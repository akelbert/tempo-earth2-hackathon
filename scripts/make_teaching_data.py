#!/usr/bin/env python
"""Create small, clearly labeled teaching datasets for the starter notebooks.

These fixtures make every notebook executable in CI and before the real case
study extracts are published. They preserve the shape, units, and joins used by
the real data, but they are synthetic and must never be presented as evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _tempo_description(
    tempo_path: str | Path,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Read only the timestamps and bounds needed to align teaching fixtures."""
    manifest_path = Path(tempo_path) / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        times = np.array(manifest["scans"], dtype="datetime64[ns]")
        west, south, east, north = (float(value) for value in manifest["bbox"])
        return times, (west, south, east, north)

    # Synthetic stores created inside tests do not carry a manifest.
    tempo = xr.open_zarr(str(tempo_path), consolidated=True)
    return tempo.time.values, (
        float(tempo.lon.min()),
        float(tempo.lat.min()),
        float(tempo.lon.max()),
        float(tempo.lat.max()),
    )


def _air_quality(
    times: np.ndarray,
    bbox: tuple[float, float, float, float],
    output: Path,
) -> None:
    lon_min, lat_min, lon_max, lat_max = bbox
    stations = [
        ("TEACH001", "Southwest teaching station", 0.25, 0.25),
        ("TEACH002", "Central teaching station", 0.52, 0.50),
        ("TEACH003", "Northeast teaching station", 0.76, 0.73),
        ("TEACH004", "Coastal teaching station", 0.36, 0.82),
    ]
    rows = []
    for station_index, (station_id, name, fy, fx) in enumerate(stations):
        latitude = lat_min + fy * (lat_max - lat_min)
        longitude = lon_min + fx * (lon_max - lon_min)
        for time_index, stamp in enumerate(times):
            urban = np.exp(-((latitude - 40.7) ** 2 + (longitude + 74.0) ** 2) / 4.0)
            column = 1.8e15 + 5.0e15 * urban + 0.3e15 * np.sin(time_index)
            hour = pd.Timestamp(stamp).hour
            pbl = 550.0 + 500.0 * np.sin(np.pi * (hour - 8) / 12) ** 2
            pbl += station_index * 45.0
            no2 = 1.7 + 1.9 * column / 1e15 * 700.0 / pbl
            no2 += 0.25 * np.sin(time_index + station_index)
            for parameter, value, unit in (
                ("NO2", no2, "Parts per billion"),
                ("O3", 30.0 + 2.8 * time_index - 0.25 * no2, "Parts per billion"),
                ("PM2.5", 6.0 + 0.35 * no2, "Micrograms/cubic meter"),
            ):
                rows.append(
                    {
                        "station_id": station_id,
                        "station_name": name,
                        "latitude": latitude,
                        "longitude": longitude,
                        "time_utc": pd.Timestamp(stamp).isoformat(),
                        "parameter": parameter,
                        "value": round(float(value), 4),
                        "unit": unit,
                        "source": "SYNTHETIC teaching fixture (not EPA observations)",
                    }
                )
    date = str(times[0])[:10]
    _write_csv(pd.DataFrame(rows), output / "aqs" / f"{date}.csv")


def _meteorology(
    time: np.ndarray,
    bbox: tuple[float, float, float, float],
    output: Path,
) -> None:
    west, south, east, north = bbox
    lat = np.linspace(south, north, 30)
    lon = np.linspace(west, east, 40)
    tt, yy, _xx = np.meshgrid(
        np.arange(len(time)), lat, lon, indexing="ij"
    )
    shape = tt.shape
    meteorology = xr.Dataset(
        {
            "pbl_height_m": (
                ("time", "lat", "lon"),
                650.0 + 350.0 * np.sin(np.pi * tt / max(1, len(time) - 1)) ** 2,
            ),
            "t2m_k": (("time", "lat", "lon"), 291.0 + 1.5 * tt + 0.08 * yy),
            "relative_humidity_pct": (
                ("time", "lat", "lon"),
                62.0 - 3.0 * tt + np.zeros(shape),
            ),
            "precip_mm_h": (
                ("time", "lat", "lon"),
                np.maximum(0.0, 1.5 - np.abs(tt - 2.0)) + np.zeros(shape),
            ),
            "u10m": (("time", "lat", "lon"), 5.5 + np.zeros(shape)),
            "v10m": (("time", "lat", "lon"), -1.5 + np.zeros(shape)),
        },
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"source": "SYNTHETIC teaching fixture (not HRRR)"},
    )
    path = output / "demo" / "meteorology.nc"
    path.parent.mkdir(parents=True, exist_ok=True)
    meteorology.to_netcdf(path, engine="h5netcdf")


def _smoke(output: Path) -> None:
    start = pd.Timestamp("2026-06-15T10:00:00Z")
    times = pd.date_range(start, periods=18, freq="h")
    stations = [
        ("SMOKE_W", "Upwind", 42.5, -78.2, 0.0),
        ("SMOKE_C", "Central", 41.5, -74.8, 4.0),
        ("SMOKE_E", "Downwind", 42.2, -71.1, 8.0),
    ]
    rows = []
    for station_id, name, latitude, longitude, delay in stations:
        for index, stamp in enumerate(times):
            plume = 46.0 * np.exp(-0.5 * ((index - 7.0 - delay / 2.0) / 2.2) ** 2)
            rows.append(
                {
                    "station_id": station_id,
                    "station_name": name,
                    "latitude": latitude,
                    "longitude": longitude,
                    "time_utc": stamp.isoformat(),
                    "pm25_ug_m3": round(5.0 + plume, 3),
                    "source": "SYNTHETIC smoke teaching fixture",
                }
            )
    _write_csv(pd.DataFrame(rows), output / "demo" / "smoke_air_quality.csv")

    fires = pd.DataFrame(
        {
            "latitude": [45.2, 44.8, 45.6, 44.5, 46.0, 43.9],
            "longitude": [-81.0, -80.4, -79.8, -78.9, -82.2, -77.8],
            "time_utc": [
                (start + pd.Timedelta(hours=hour)).isoformat()
                for hour in (0, 1, 1, 2, 3, 4)
            ],
            "frp_mw": [34.0, 82.0, 45.0, 67.0, 29.0, 51.0],
            "source": ["SYNTHETIC fire detection"] * 6,
        }
    )
    _write_csv(fires, output / "demo" / "fire_detections.csv")


def _wetland(output: Path) -> None:
    rng = np.random.default_rng(20260914)
    dates = pd.date_range("2026-04-01", periods=90, freq="D", tz="UTC")
    rows = []
    for site_index, site in enumerate(("MARSH_A", "MARSH_B", "MARSH_C", "MARSH_D")):
        phase = site_index * 0.7
        water = 0.35 + 0.18 * np.sin(np.arange(len(dates)) / 8.0 + phase)
        storm = np.exp(-0.5 * ((np.arange(len(dates)) - 48) / 3.5) ** 2)
        water = water + 0.42 * storm + rng.normal(0, 0.018, len(dates))
        salinity = 15.0 - 8.0 * water + site_index + rng.normal(0, 0.35, len(dates))
        redox = 260.0 - 330.0 * water + rng.normal(0, 12.0, len(dates))
        temperature = 13.0 + 8.0 * np.sin(np.arange(len(dates)) / 30.0)
        carbon = 0.8 + 0.035 * temperature - 0.002 * redox
        carbon += rng.normal(0, 0.08, len(dates))
        for index, stamp in enumerate(dates):
            rows.append(
                {
                    "site_id": site,
                    "latitude": 38.86 + 0.015 * site_index,
                    "longitude": -76.55 - 0.018 * site_index,
                    "date": stamp.date().isoformat(),
                    "water_level_m": round(float(water[index]), 4),
                    "salinity_psu": round(float(salinity[index]), 4),
                    "redox_mv": round(float(redox[index]), 3),
                    "temperature_c": round(float(temperature[index]), 3),
                    "carbon_flux_umol_m2_s": round(float(carbon[index]), 5),
                    "source": "SYNTHETIC wetland teaching fixture",
                }
            )
    wetland = pd.DataFrame(rows)
    _write_csv(wetland, output / "demo" / "wetland_timeseries.csv")
    _write_csv(
        wetland.query("site_id == 'MARSH_A'").drop(columns="source"),
        output / "demo" / "site_timeseries.csv",
    )


def _land_surface(output: Path) -> None:
    lat = np.linspace(38.72, 39.08, 180)
    lon = np.linspace(-76.78, -76.30, 240)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    forest = np.exp(-((yy - 38.91) ** 2 + (xx + 76.55) ** 2) / 0.018)
    river = np.exp(-((xx + 76.43 + 0.10 * np.sin((yy - 38.8) * 20)) / 0.012) ** 2)
    urban = np.exp(-((yy - 38.84) ** 2 + (xx + 76.68) ** 2) / 0.003)
    red = 0.16 - 0.07 * forest + 0.04 * urban - 0.05 * river
    nir = 0.24 + 0.34 * forest - 0.07 * urban - 0.16 * river
    red_edge = 0.20 + 0.18 * forest - 0.04 * urban - 0.10 * river
    scene = xr.Dataset(
        {
            "red": (("lat", "lon"), red.astype("float32")),
            "nir": (("lat", "lon"), nir.astype("float32")),
            "red_edge": (("lat", "lon"), red_edge.astype("float32")),
        },
        coords={"lat": lat, "lon": lon},
        attrs={"source": "SYNTHETIC surface-reflectance teaching fixture (not HLS)"},
    )
    path = output / "demo" / "land_surface.nc"
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.to_netcdf(path, engine="h5netcdf")


def build(tempo_path: str | Path, output: str | Path) -> Path:
    """Write the complete teaching fixture collection and return its root."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    times, bbox = _tempo_description(tempo_path)
    print("  air-quality fixture", flush=True)
    _air_quality(times, bbox, root)
    print("  meteorology fixture", flush=True)
    _meteorology(times, bbox, root)
    print("  smoke fixture", flush=True)
    _smoke(root)
    print("  wetland fixture", flush=True)
    _wetland(root)
    print("  multiscale fixture", flush=True)
    _land_surface(root)
    manifest = {
        "title": "Living Earth Twin workshop teaching fixtures",
        "synthetic": True,
        "warning": "For instruction and automated tests only; not scientific evidence.",
        "datasets": [
            "aqs/<tempo-date>.csv",
            "demo/meteorology.nc",
            "demo/smoke_air_quality.csv",
            "demo/fire_detections.csv",
            "demo/wetland_timeseries.csv",
            "demo/site_timeseries.csv",
            "demo/land_surface.nc",
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tempo", required=True, help="staged TEMPO Zarr store")
    parser.add_argument("--output", default="data/context", help="output root")
    args = parser.parse_args()
    root = build(args.tempo, args.output)
    print(f"Wrote teaching context data to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
