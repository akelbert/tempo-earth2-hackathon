#!/usr/bin/env python
"""Stage EPA AQS hourly air-quality observations for a bounded exploration case.

The national annual archives are downloaded once into a local cache, streamed
from their ZIP files, filtered to a UTC date range and workshop region, and
reduced to one mean value per station, pollutant, and hour. Use ``--date`` for
a notebook-sized case or ``--start-date``/``--end-date`` for a broader event
collection. CSV.gz is recommended for the latter.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import sys
import urllib.request
import zipfile
from collections import defaultdict
from datetime import UTC, date, datetime
from itertools import groupby
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tempo_earth2.tempo import REGIONS

PARAMETERS = {
    "NO2": "42602",
    "O3": "44201",
    "PM2.5": "88101",
}
BASE_URL = "https://aqs.epa.gov/aqsweb/airdata"


def _download(parameter: str, year: int, cache: Path) -> tuple[Path, str]:
    code = PARAMETERS[parameter]
    url = f"{BASE_URL}/hourly_{code}_{year}.zip"
    target = cache / f"hourly_{code}_{year}.zip"
    if target.is_file() and target.stat().st_size:
        return target, url
    cache.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {parameter}: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "tempo-earth2-workshop/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        target.write_bytes(response.read())
    return target, url


def _read(
    archive: Path,
    parameter: str,
    start_date: str,
    end_date: str,
    bbox: tuple[float, float, float, float],
) -> tuple[list[dict[str, object]], tuple[str | None, str | None]]:
    west, south, east, north = bbox
    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    first_date: str | None = None
    last_date: str | None = None
    with zipfile.ZipFile(archive) as zipped:
        csv_name = next(name for name in zipped.namelist() if name.endswith(".csv"))
        with zipped.open(csv_name) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            for row in csv.DictReader(text):
                row_date = row["Date GMT"]
                first_date = row_date if first_date is None else min(first_date, row_date)
                last_date = row_date if last_date is None else max(last_date, row_date)
                if not start_date <= row_date <= end_date:
                    continue
                latitude = float(row["Latitude"])
                longitude = float(row["Longitude"])
                if not (west <= longitude <= east and south <= latitude <= north):
                    continue
                try:
                    value = float(row["Sample Measurement"])
                except ValueError:
                    continue
                station_id = "-".join(
                    (row["State Code"], row["County Code"], row["Site Num"])
                )
                key = (
                    station_id,
                    row.get("Local Site Name", station_id),
                    latitude,
                    longitude,
                    f"{row['Date GMT']}T{row['Time GMT']}:00Z",
                    parameter,
                    row["Units of Measure"],
                )
                grouped[key].append(value)

    records = []
    for key, values in grouped.items():
        station_id, name, latitude, longitude, stamp, pollutant, unit = key
        records.append(
            {
                "station_id": station_id,
                "station_name": name,
                "latitude": latitude,
                "longitude": longitude,
                "time_utc": stamp,
                "parameter": pollutant,
                "value": round(sum(values) / len(values), 6),
                "unit": unit,
                "source": "US EPA Air Quality System (AQS)",
            }
        )
    return records, (first_date, last_date)


def _write_csv(handle, fields: list[str], records: list[dict[str, object]]) -> None:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    dates = parser.add_mutually_exclusive_group(required=True)
    dates.add_argument("--date", help="one UTC date, YYYY-MM-DD")
    dates.add_argument("--start-date", help="first UTC date, YYYY-MM-DD")
    parser.add_argument(
        "--end-date",
        help="last UTC date, YYYY-MM-DD; required with --start-date",
    )
    parser.add_argument("--region", choices=sorted(REGIONS), default="northeast")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--partition-by-month",
        action="store_true",
        help="treat --output as a directory and write one YYYY-MM.csv.gz per month",
    )
    parser.add_argument("--cache", type=Path, default=Path(".cache/aqs"))
    parser.add_argument(
        "--parameters", nargs="+", choices=sorted(PARAMETERS), default=list(PARAMETERS)
    )
    args = parser.parse_args()

    if args.date:
        start_date = end_date = args.date
    else:
        if not args.end_date:
            parser.error("--end-date is required with --start-date")
        start_date, end_date = args.start_date, args.end_date
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except ValueError as exc:
        parser.error(f"dates must use YYYY-MM-DD: {exc}")
    if end < start:
        parser.error("--end-date must not be earlier than --start-date")

    region = REGIONS[args.region]
    records = []
    sources = []
    coverage = {}
    for year in range(start.year, end.year + 1):
        year_start = max(start_date, f"{year}-01-01")
        year_end = min(end_date, f"{year}-12-31")
        for parameter in args.parameters:
            archive, url = _download(parameter, year, args.cache)
            sources.append(url)
            parameter_records, date_range = _read(
                archive, parameter, year_start, year_end, region.bbox
            )
            records.extend(parameter_records)
            coverage[f"{parameter}-{year}"] = {
                "first_utc": date_range[0],
                "last_utc": date_range[1],
            }
    records.sort(key=lambda row: (row["time_utc"], row["parameter"], row["station_id"]))
    if not records:
        available = ", ".join(
            f"{name}: {dates['first_utc']} through {dates['last_utc']}"
            for name, dates in coverage.items()
        )
        raise SystemExit(
            "No AQS observations matched the requested date and region. "
            f"Current archive coverage is {available}."
        )

    fields = list(records[0])
    if args.partition_by_month:
        args.output.mkdir(parents=True, exist_ok=True)
        for month, monthly_records in groupby(
            records, key=lambda record: str(record["time_utc"])[:7]
        ):
            partition = args.output / f"{month}.csv.gz"
            with gzip.open(partition, "wt", newline="") as handle:
                _write_csv(handle, fields, monthly_records)
            print(f"Wrote {partition}")
        manifest_path = args.output / "manifest.json"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix == ".gz":
            with gzip.open(args.output, "wt", newline="") as handle:
                _write_csv(handle, fields, records)
        else:
            with args.output.open("w", newline="") as handle:
                _write_csv(handle, fields, records)
        manifest_path = args.output.with_suffix(".manifest.json")

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "US EPA Air Quality System (AQS)",
        "source_urls": sorted(set(sources)),
        "start_date_utc": start_date,
        "end_date_utc": end_date,
        "region": args.region,
        "bbox": region.bbox,
        "parameters": args.parameters,
        "source_archive_coverage": coverage,
        "rows": len(records),
        "layout": "monthly CSV.gz partitions" if args.partition_by_month else "single CSV",
        "processing": "Mean of duplicate POC/method observations by station and UTC hour.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(records):,} observations under {args.output}")
    print(f"Wrote provenance to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
