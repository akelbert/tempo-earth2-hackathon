#!/usr/bin/env python
"""Stage a bounded NOAA HMS + AirNow smoke-event extract.

The output is deliberately small enough for an introductory notebook:

* analyst-drawn NOAA HMS smoke polygons for one UTC day;
* NOAA HMS satellite fire detections in the source/transport region; and
* AirNow daily PM2.5 observations for a short window.

AirNow observations are preliminary and are not a replacement for regulatory
EPA AQS data.  The generated manifest records that distinction and every source
URL used to produce the extract.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import urllib.request
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import shapefile
from shapely.geometry import Polygon, box, mapping
from shapely.ops import unary_union

DEFAULT_BBOX = (-100.0, 35.0, -65.0, 56.0)
AIRNOW_FIELDS = (
    "valid_date_local",
    "station_id",
    "station_name",
    "parameter",
    "units",
    "value",
    "averaging_period_hours",
    "data_source",
    "aqi",
    "aqi_category",
    "latitude",
    "longitude",
    "full_station_id",
)


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "tempo-earth2-workshop"})
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())


def _archive_url(kind: str, day: date) -> str:
    root = "https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS"
    folder = "Smoke_Polygons" if kind == "smoke" else "Fire_Points"
    return (
        f"{root}/{folder}/Shapefile/{day:%Y}/{day:%m}/"
        f"hms_{kind}{day:%Y%m%d}.zip"
    )


def _reader(archive: Path, workspace: Path) -> shapefile.Reader:
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(workspace)
    return shapefile.Reader(next(workspace.glob("*.shp")))


def _stage_smoke(reader: shapefile.Reader, bounds) -> dict:
    clip = box(*bounds)
    features = []
    for item in reader.iterShapeRecords():
        starts = list(item.shape.parts) + [len(item.shape.points)]
        rings = [
            Polygon(item.shape.points[starts[index] : starts[index + 1]])
            for index in range(len(starts) - 1)
            if starts[index + 1] - starts[index] >= 4
        ]
        geometry = unary_union([ring.buffer(0) for ring in rings if not ring.is_empty])
        if not geometry.intersects(clip):
            continue
        geometry = geometry.intersection(clip)
        if geometry.is_empty:
            continue
        properties = item.record.as_dict()
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(geometry),
                "properties": {
                    "satellite": properties["Satellite"],
                    "start_utc": properties["Start"],
                    "end_utc": properties["End"],
                    "density": properties["Density"],
                    "source": "NOAA Hazard Mapping System",
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _stage_fires(reader: shapefile.Reader, bounds) -> list[dict]:
    west, south, east, north = bounds
    rows = []
    for item in reader.iterShapeRecords():
        record = item.record.as_dict()
        lon, lat = float(record["Lon"]), float(record["Lat"])
        if not (west <= lon <= east and south <= lat <= north):
            continue
        year_day = str(record["YearDay"])
        observed = datetime.strptime(
            f"{year_day}{str(record['Time']).zfill(4)}", "%Y%j%H%M"
        ).replace(tzinfo=UTC)
        rows.append(
            {
                "time_utc": observed.isoformat().replace("+00:00", "Z"),
                "longitude": lon,
                "latitude": lat,
                "satellite": record["Satellite"],
                "method": record["Method"],
                "ecosystem": record["Ecosystem"],
                "frp_mw": record["FRP"],
                "source": "NOAA Hazard Mapping System",
            }
        )
    return rows


def _stage_airnow(start: date, end: date, bounds) -> tuple[list[dict], list[str]]:
    west, south, east, north = bounds
    rows, urls = [], []
    day = start
    while day <= end:
        url = (
            f"https://files.airnowtech.org/airnow/{day:%Y}/{day:%Y%m%d}/"
            "daily_data_v2.dat"
        )
        urls.append(url)
        request = urllib.request.Request(
            url, headers={"User-Agent": "tempo-earth2-workshop"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            lines = (line.decode("utf-8") for line in response)
            for values in csv.reader(lines, delimiter="|"):
                if len(values) != len(AIRNOW_FIELDS):
                    continue
                row = dict(zip(AIRNOW_FIELDS, values, strict=True))
                if row["parameter"].upper() != "PM2.5-24HR":
                    continue
                lat, lon = float(row["latitude"]), float(row["longitude"])
                value = float(row["value"])
                if west <= lon <= east and south <= lat <= north and value >= 0:
                    row.update(
                        latitude=lat,
                        longitude=lon,
                        value=value,
                        source="AirNow preliminary observations",
                    )
                    rows.append(row)
        day += timedelta(days=1)
    return rows, urls


def _write_csv(rows: list[dict], destination: Path) -> None:
    if not rows:
        raise RuntimeError(f"No rows selected for {destination}")
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-date", type=date.fromisoformat, required=True)
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.end_date < args.start_date:
        parser.error("--end-date must not precede --start-date")
    args.output.mkdir(parents=True, exist_ok=True)

    smoke_url = _archive_url("smoke", args.event_date)
    fire_url = _archive_url("fire", args.event_date)
    with tempfile.TemporaryDirectory(prefix="smoke-stage-") as temporary:
        workspace = Path(temporary)
        smoke_zip, fire_zip = workspace / "smoke.zip", workspace / "fire.zip"
        _download(smoke_url, smoke_zip)
        _download(fire_url, fire_zip)
        smoke = _stage_smoke(_reader(smoke_zip, workspace / "smoke"), args.bbox)
        fires = _stage_fires(_reader(fire_zip, workspace / "fires"), args.bbox)

    air, airnow_urls = _stage_airnow(args.start_date, args.end_date, args.bbox)
    smoke_path = args.output / "hms_smoke.geojson"
    fire_path = args.output / "hms_fires.csv"
    air_path = args.output / "airnow_pm25_daily.csv"
    smoke_path.write_text(json.dumps(smoke, indent=2) + "\n")
    _write_csv(fires, fire_path)
    _write_csv(air, air_path)

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "event_date_utc": args.event_date.isoformat(),
        "analysis_window": [args.start_date.isoformat(), args.end_date.isoformat()],
        "bbox": list(args.bbox),
        "sources": {
            "smoke": smoke_url,
            "fires": fire_url,
            "air_quality": airnow_urls,
        },
        "rows": {
            "smoke_polygons": len(smoke["features"]),
            "fire_detections": len(fires),
            "airnow_daily_pm25": len(air),
        },
        "caveats": [
            "HMS polygons are analyst interpretations of visible satellite smoke, not surface concentrations.",
            "AirNow observations are preliminary and not regulatory AQS data.",
            "Coincidence of smoke, fire, PM2.5, wind, and TEMPO fields does not by itself establish attribution.",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["rows"], indent=2))


if __name__ == "__main__":
    main()
