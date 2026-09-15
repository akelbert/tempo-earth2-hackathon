#!/usr/bin/env python
"""Stage the NO2 forecast intercomparison case used by notebook 10.

Notebook 10 compares TEMPO NO2 over the Northeast on one day against

* transport: TEMPO advected with Earth-2 (FCN, run live on the Hub), NOAA GFS
  forecast winds and ERA5 reanalysis winds;
* chemistry forecasts: CAMS global forecast NO2 (ECMWF), NASA GEOS-CF v2
  forecast surface NO2 and GEOS-CF v2 analysis NO2 columns;
* the surface: EPA AirNow hourly NO2 monitors, and optionally the
  OpenWeatherMap air-pollution history at those monitors.

Every source is cut to a padded Northeast box and a few hours so the staged
case is a few megabytes and can live in the repository. Nothing here is
resampled in space: the notebook does the regridding where it can be seen.

Beyond ``make venv``, staging needs ``pip install cfgrib eccodes cdsapi``
(GFS GRIB decoding and the ADS client). The notebook itself needs neither.

Public, no credentials:   geoscf, gfs, era5, airnow
Needs an ADS API key:     cams  (~/.ecmwfdatastoresrc or ~/.cdsapirc, see
                                 https://ads.atmosphere.copernicus.eu/how-to-api)
Needs an OpenWeatherMap key: owm (OPENWEATHERMAP_API_KEY environment variable)

    python scripts/stage_no2_intercomparison.py                # public sources
    python scripts/stage_no2_intercomparison.py --only cams owm # keyed sources
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
USER_AGENT = {"User-Agent": "tempo-earth2-workshop"}

#: Northeast region from tempo_earth2.tempo.REGIONS padded by 1 degree, so the
#: notebook can regrid and advect without an artificial edge at the TEMPO box.
DEFAULT_BBOX = (-81.0, 35.0, -65.0, 46.0)  # west, south, east, north

#: Hours (UTC) bracketing the staged TEMPO scans (14:09-19:09Z on 2026-06-15).
DEFAULT_HOURS = tuple(range(13, 21))

GEOSCF_ROOT = "https://portal.nccs.nasa.gov/datashare/gmao/geos-cf/v2"
GFS_ROOT = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
ERA5_ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
AIRNOW_ROOT = "https://files.airnowtech.org/airnow"
OWM_HISTORY = "https://api.openweathermap.org/data/2.5/air_pollution/history"
CAMS_DATASET = "cams-global-atmospheric-composition-forecasts"

AVOGADRO = 6.02214076e23
NO2_MOLAR_MASS_KG = 0.0460055
DRY_AIR_MOLAR_MASS_KG = 0.0289644


# --------------------------------------------------------------------------- io


def _utc(day: date, hour: int, minute: int = 0) -> datetime:
    """A UTC wall-clock time as a naive datetime, matching xarray time coordinates."""
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC).replace(tzinfo=None)


def _fetch(url: str, attempts: int = 3, headers: dict | None = None) -> bytes:
    request = urllib.request.Request(url, headers={**USER_AGENT, **(headers or {})})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.read()
        except OSError:
            if attempt == attempts:
                raise
            time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def _crop(ds: xr.Dataset, bbox, lat="lat", lon="lon") -> xr.Dataset:
    """Crop to bbox with longitude in -180..180 and latitude ascending."""
    west, south, east, north = bbox
    if float(ds[lon].max()) > 180.0:
        ds = ds.assign_coords({lon: ((ds[lon] + 180.0) % 360.0) - 180.0}).sortby(lon)
    if ds[lat].size > 1 and ds[lat].values[0] > ds[lat].values[-1]:
        ds = ds.sortby(lat)
    return ds.sel({lat: slice(south, north), lon: slice(west, east)})


def _write_netcdf(ds: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        name: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for name in ds.data_vars
    }
    ds.to_netcdf(path, engine="netcdf4", encoding=encoding)
    print(f"  wrote {path.relative_to(ROOT)}  ({path.stat().st_size / 1e6:.2f} MB)")


# ---------------------------------------------------------------------- GEOS-CF


def _geoscf_file(kind: str, collection: str, day: date, valid: datetime, run: str) -> str:
    folder = f"{GEOSCF_ROOT}/{kind}/Y{day:%Y}/M{day:%m}/D{day:%d}"
    stamp = f"{valid:%Y%m%d_%H%M}z"
    if kind == "fcst":
        name = f"GEOS.cf.fcst.{collection}.{day:%Y%m%d}_{run}+{stamp}.R0.nc4"
        name = name.replace("+", "%2B")
    else:
        name = f"GEOS.cf.ana.{collection}.{stamp}.R0.nc4"
    return f"{folder}/{name}"


def _geoscf_series(kind, collection, variables, day, hours, bbox, run="09z"):
    frames, urls = [], []
    with tempfile.TemporaryDirectory(prefix="geoscf-") as workspace:
        for hour in hours:
            # GEOS-CF 1-hourly time-averaged files are stamped at the half hour.
            valid = _utc(day, hour, 30)
            url = _geoscf_file(kind, collection, day, valid, run)
            local = Path(workspace) / f"{kind}_{collection}_{hour:02d}.nc4"
            print(f"  GEOS-CF {kind} {collection} {valid:%H:%M}Z")
            local.write_bytes(_fetch(url))
            with xr.open_dataset(local) as ds:
                cut = _crop(ds[list(variables)], bbox).load()
            if "lev" in cut.dims:
                cut = cut.isel(lev=0, drop=True)
            frames.append(cut)
            urls.append(url)
    return xr.concat(frames, dim="time"), urls


def stage_geoscf(day: date, hours, bbox, out: Path) -> dict:
    """GEOS-CF v2: forecast surface NO2 and analysis NO2 columns."""
    fcst, fcst_urls = _geoscf_series(
        "fcst", "aqc_tavg_1hr_glo_L1440x721_slv", ("NO2",), day, hours, bbox
    )
    fcst = fcst.rename(NO2="no2_surface")
    fcst["no2_surface"] = fcst["no2_surface"] * 1e9
    fcst["no2_surface"].attrs = {
        "long_name": "GEOS-CF forecast NO2 dry-air mole fraction, lowest model layer",
        "units": "ppbv",
    }
    fcst.attrs = {
        "title": "NASA GEOS-CF v2 forecast, surface NO2 (Northeast subset)",
        "source": "NASA GMAO GEOS-CF v2 near-real-time forecast (aqc collection)",
        "forecast_run": f"{day:%Y-%m-%d} 09z run tag; files generated ~12Z, before the first TEMPO scan",
        "time_convention": "1-hour means centred on the time stamp",
        "caveat": "GEOS-CF assimilates satellite observations, so it is not fully independent of satellite NO2.",
    }
    _write_netcdf(fcst, out / "geoscf_fcst_surface_no2.nc")

    ana, ana_urls = _geoscf_series(
        "ana",
        "xgc_tavg_1hr_glo_L1440x721_slv",
        ("TropCol_NO2", "TotCol_NO2", "PBLCol_NO2"),
        day,
        hours,
        bbox,
    )
    ana = ana.rename(
        TropCol_NO2="no2_trop", TotCol_NO2="no2_total", PBLCol_NO2="no2_pbl"
    )
    for name in ana.data_vars:
        # Stored as 1e15 molecules cm-2; convert to TEMPO's molecules cm-2.
        long_name = ana[name].attrs.get("long_name", name)
        ana[name] = ana[name] * 1e15
        ana[name].attrs = {"long_name": long_name, "units": "molecules/cm^2"}
    ana.attrs = {
        "title": "NASA GEOS-CF v2 analysis, NO2 columns (Northeast subset)",
        "source": "NASA GMAO GEOS-CF v2 near-real-time analysis (xgc collection)",
        "time_convention": "1-hour means centred on the time stamp",
        "caveat": (
            "An observation-constrained analysis, not a forecast. The forecast "
            "column collection is purged from the public portal after ~2 weeks."
        ),
    }
    _write_netcdf(ana, out / "geoscf_ana_no2_columns.nc")
    return {
        "geoscf_fcst_surface_no2.nc": {"urls": fcst_urls},
        "geoscf_ana_no2_columns.nc": {"urls": ana_urls},
    }


# -------------------------------------------------------------------------- GFS

_GFS_FIELDS = {
    ("UGRD", "10 m above ground"): "u10m",
    ("VGRD", "10 m above ground"): "v10m",
    ("UGRD", "100 m above ground"): "u100m",
    ("VGRD", "100 m above ground"): "v100m",
    ("UGRD", "850 mb"): "u850",
    ("VGRD", "850 mb"): "v850",
}


def _gfs_messages(grib_url: str) -> bytes:
    """Byte-range only the six wind messages out of a 500 MB GFS file."""
    index = _fetch(grib_url + ".idx").decode().strip().splitlines()
    entries = [line.split(":") for line in index]
    chunks = []
    for i, entry in enumerate(entries):
        if (entry[3], entry[4]) not in _GFS_FIELDS:
            continue
        start = int(entry[1])
        end = int(entries[i + 1][1]) - 1 if i + 1 < len(entries) else ""
        chunks.append(_fetch(grib_url, headers={"Range": f"bytes={start}-{end}"}))
    if len(chunks) != len(_GFS_FIELDS):
        raise RuntimeError(f"Expected {len(_GFS_FIELDS)} wind messages in {grib_url}")
    return b"".join(chunks)


def stage_gfs(day: date, hours, bbox, out: Path, init_hour: int = 12) -> dict:
    """NOAA GFS 0.25-degree forecast winds, hourly lead times, from AWS."""
    import warnings

    import cfgrib

    # cfgrib's internal xr.merge warns about a future xarray default it does
    # not control; the six single-level messages here cannot conflict.
    warnings.filterwarnings("ignore", category=FutureWarning, module="cfgrib")

    init = _utc(day, init_hour)
    frames, urls = [], []
    with tempfile.TemporaryDirectory(prefix="gfs-") as workspace:
        for hour in hours:
            valid = _utc(day, hour)
            lead = int((valid - init).total_seconds() // 3600)
            if lead < 0:
                continue
            url = (
                f"{GFS_ROOT}/gfs.{init:%Y%m%d}/{init:%H}/atmos/"
                f"gfs.t{init:%H}z.pgrb2.0p25.f{lead:03d}"
            )
            print(f"  GFS {init:%H}Z +{lead}h (valid {valid:%H}Z)")
            local = Path(workspace) / f"gfs_f{lead:03d}.grib2"
            local.write_bytes(_gfs_messages(url))
            fields = {}
            for ds in cfgrib.open_datasets(str(local), backend_kwargs={"indexpath": ""}):
                level_type = next(
                    c for c in ("heightAboveGround", "isobaricInhPa") if c in ds.coords
                )
                level = int(ds[level_type].values)
                suffix = f"{level}m" if level_type == "heightAboveGround" else f"{level}"
                for name in ds.data_vars:
                    fields[f"{name[0]}{suffix}"] = ds[name].drop_vars(
                        [c for c in ds[name].coords if c not in ("latitude", "longitude")]
                    )
            merged = xr.Dataset(fields).rename(latitude="lat", longitude="lon")
            merged = _crop(merged, bbox).load()
            frames.append(merged.expand_dims(time=[np.datetime64(valid, "ns")]))
            urls.append(url)
    gfs = xr.concat(frames, dim="time")
    gfs = gfs[sorted(gfs.data_vars)]
    for name in gfs.data_vars:
        gfs[name].attrs = {"units": "m s-1", "long_name": f"GFS forecast {name}"}
    gfs = gfs.assign_coords(
        init_time=np.datetime64(init, "ns"),
        lead_hours=("time", [(pd.Timestamp(t) - init).total_seconds() / 3600 for t in gfs.time.values]),
    )
    gfs.attrs = {
        "title": "NOAA GFS 0.25-degree forecast winds (Northeast subset)",
        "source": "NOAA Open Data Dissemination, noaa-gfs-bdp-pds",
        "forecast_init": f"{init:%Y-%m-%dT%H:%M}Z",
        "note": "Instantaneous winds at the valid hour; FCN is initialized from this GFS cycle.",
    }
    _write_netcdf(gfs, out / "gfs_fcst_winds.nc")
    return {"gfs_fcst_winds.nc": {"urls": urls, "forecast_init": gfs.attrs["forecast_init"]}}


# ------------------------------------------------------------------------- ERA5


def stage_era5(day: date, hours, bbox, out: Path) -> dict:
    """ERA5 (ERA5T for recent months) hourly winds from the public ARCO store."""
    store = xr.open_zarr(
        ERA5_ARCO, chunks=None, storage_options={"token": "anon"}, consolidated=True
    )
    times = [np.datetime64(_utc(day, h), "ns") for h in hours]
    west, south, east, north = bbox
    lon_slice = slice(west % 360.0, east % 360.0)
    print(f"  ERA5 {len(times)} hours from {ERA5_ARCO}")
    surface = store[
        [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "100m_u_component_of_wind",
            "100m_v_component_of_wind",
        ]
    ].sel(time=times, latitude=slice(north, south), longitude=lon_slice)
    upper = store[["u_component_of_wind", "v_component_of_wind"]].sel(
        time=times, level=850, latitude=slice(north, south), longitude=lon_slice
    )
    era5 = xr.Dataset(
        {
            "u10m": surface["10m_u_component_of_wind"],
            "v10m": surface["10m_v_component_of_wind"],
            "u100m": surface["100m_u_component_of_wind"],
            "v100m": surface["100m_v_component_of_wind"],
            "u850": upper["u_component_of_wind"].drop_vars("level"),
            "v850": upper["v_component_of_wind"].drop_vars("level"),
        }
    ).rename(latitude="lat", longitude="lon")
    era5 = _crop(era5.load(), bbox)
    for name in era5.data_vars:
        era5[name].attrs = {"units": "m s-1", "long_name": f"ERA5 {name}"}
    era5.attrs = {
        "title": "ERA5 hourly reanalysis winds (Northeast subset)",
        "source": "ECMWF ERA5 via Google ARCO-ERA5",
        "valid_time_stop_final_era5": store.attrs.get("valid_time_stop", ""),
        "note": (
            "Reanalysis, not a forecast: it has seen observations after the scan. "
            "Months after valid_time_stop are preliminary ERA5T."
        ),
    }
    _write_netcdf(era5, out / "era5_winds.nc")
    return {"era5_winds.nc": {"urls": [ERA5_ARCO]}}


# ----------------------------------------------------------------------- AirNow


def _airnow_sites(day: date) -> pd.DataFrame:
    url = f"{AIRNOW_ROOT}/{day:%Y}/{day:%Y%m%d}/Monitoring_Site_Locations_V2.dat"
    table = pd.read_csv(io.BytesIO(_fetch(url)), sep="|", dtype=str)
    table = table.loc[table["Parameter"] == "NO2"]
    sites = pd.concat(
        [
            table.assign(station_id=table["AQSID"]),
            table.assign(station_id=table["FullAQSID"]),
        ]
    ).drop_duplicates("station_id")
    sites["latitude"] = sites["Latitude"].astype(float)
    sites["longitude"] = sites["Longitude"].astype(float)
    return sites[["station_id", "SiteName", "latitude", "longitude", "AgencyName"]], url


def stage_airnow(day: date, bbox, out: Path) -> dict:
    """Preliminary AirNow hourly NO2 for every monitor in the box, all day."""
    west, south, east, north = bbox
    sites, sites_url = _airnow_sites(day)
    rows, urls = [], [sites_url]
    for hour in range(24):
        url = f"{AIRNOW_ROOT}/{day:%Y}/{day:%Y%m%d}/HourlyData_{day:%Y%m%d}{hour:02d}.dat"
        urls.append(url)
        text = _fetch(url).decode("utf-8", errors="replace")
        for values in csv.reader(io.StringIO(text), delimiter="|"):
            if len(values) < 8 or values[5] != "NO2":
                continue
            rows.append(
                {
                    "station_id": values[2],
                    "station_name": values[3],
                    # HourlyData files are keyed by UTC hour; values are
                    # hour-beginning averages.
                    "time_utc": f"{day:%Y-%m-%d}T{hour:02d}:00:00Z",
                    "value": float(values[7]),
                    "unit": values[6],
                    "agency": values[8] if len(values) > 8 else "",
                }
            )
    frame = pd.DataFrame(rows).merge(
        sites[["station_id", "latitude", "longitude"]], on="station_id", how="inner"
    )
    frame = frame.loc[
        frame.longitude.between(west, east) & frame.latitude.between(south, north)
    ]
    frame["parameter"] = "NO2"
    frame["source"] = "EPA AirNow hourly (preliminary, not regulatory AQS)"
    frame = frame[
        ["station_id", "station_name", "latitude", "longitude", "time_utc",
         "parameter", "value", "unit", "agency", "source"]
    ].sort_values(["station_id", "time_utc"])
    path = out / "airnow_no2_hourly.csv"
    frame.to_csv(path, index=False)
    print(f"  wrote {path.relative_to(ROOT)}  ({len(frame):,} rows, "
          f"{frame.station_id.nunique()} stations)")
    return {"airnow_no2_hourly.csv": {"urls": urls, "rows": len(frame)}}


# ------------------------------------------------------------------------- CAMS


def stage_cams(day: date, hours, bbox, out: Path, init_hour: int = 0) -> dict:
    """CAMS global forecast NO2: hourly total column, 3-hourly lowest level."""
    import cdsapi

    west, south, east, north = bbox
    init = _utc(day, init_hour)
    leads = [int((_utc(day, h) - init).total_seconds() // 3600) for h in hours]
    # Lowest-model-level fields are only issued every 3 hours; bracket the scans.
    leads_3h = sorted({3 * (lead // 3) for lead in leads} | {3 * (lead // 3) + 3 for lead in leads})
    client = cdsapi.Client(url=os.environ.get("ADS_URL", "https://ads.atmosphere.copernicus.eu/api"))
    common = {
        "date": [f"{day:%Y-%m-%d}/{day:%Y-%m-%d}"],
        "type": ["forecast"],
        "time": [f"{init_hour:02d}:00"],
        "data_format": "netcdf",
        "area": [north, west, south, east],
    }
    requests = {
        "column": {**common, "variable": ["total_column_nitrogen_dioxide"],
                   "leadtime_hour": [str(x) for x in leads]},
        "surface": {**common, "variable": ["nitrogen_dioxide"], "model_level": ["137"],
                    "leadtime_hour": [str(x) for x in leads_3h]},
    }
    parts = {}
    with tempfile.TemporaryDirectory(prefix="cams-") as workspace:
        for label, body in requests.items():
            target = Path(workspace) / f"{label}.nc"
            print(f"  CAMS {label} request (init {init:%H}Z, leads {body['leadtime_hour']})")
            client.retrieve(CAMS_DATASET, body, str(target))
            ds = xr.open_dataset(target, decode_timedelta=False).load()
            parts[label] = ds

    def _tidy(ds: xr.Dataset) -> xr.Dataset:
        ds = ds.rename({k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in ds.dims})
        if "valid_time" in ds.coords and "forecast_period" in ds.dims:
            ds = ds.swap_dims(forecast_period="valid_time")
        drop = [d for d in ds.dims if d not in ("valid_time", "lat", "lon") and ds.sizes[d] == 1]
        ds = ds.isel({d: 0 for d in drop}, drop=True)
        ds = ds.rename(valid_time="time")
        return _crop(ds, bbox)

    column = _tidy(parts["column"])
    surface = _tidy(parts["surface"])
    column_name = next(iter(column.data_vars))
    surface_name = next(iter(surface.data_vars))
    to_molecules_cm2 = AVOGADRO / NO2_MOLAR_MASS_KG / 1e4
    to_ppbv = DRY_AIR_MOLAR_MASS_KG / NO2_MOLAR_MASS_KG * 1e9
    cams = xr.Dataset(
        {
            "no2_total": column[column_name] * to_molecules_cm2,
            "no2_surface": (surface[surface_name] * to_ppbv).rename(time="time_3h"),
        }
    )
    cams["no2_total"].attrs = {"long_name": "CAMS forecast total column NO2", "units": "molecules/cm^2"}
    cams["no2_surface"].attrs = {
        "long_name": "CAMS forecast NO2 mole fraction, lowest model level (137)",
        "units": "ppbv",
    }
    cams.attrs = {
        "title": "CAMS global atmospheric composition forecast NO2 (Northeast subset)",
        "source": f"Copernicus Atmosphere Data Store, {CAMS_DATASET}",
        "forecast_init": f"{init:%Y-%m-%dT%H:%M}Z",
        "grid": "0.4 degree",
        "caveat": "CAMS assimilates satellite NO2, so it is not fully independent of satellite NO2.",
    }
    _write_netcdf(cams, out / "cams_fcst_no2.nc")
    return {"cams_fcst_no2.nc": {"requests": requests, "forecast_init": cams.attrs["forecast_init"]}}


# ------------------------------------------------------------------ OpenWeather


def stage_owm(day: date, out: Path, box=(-74.3, 40.5, -73.7, 41.0)) -> dict:
    """OpenWeatherMap air-pollution history at the NYC-area AirNow NO2 monitors."""
    key = os.environ.get("OPENWEATHERMAP_API_KEY")
    if not key:
        raise SystemExit("Set OPENWEATHERMAP_API_KEY to stage OpenWeatherMap history")
    airnow = pd.read_csv(out / "airnow_no2_hourly.csv")
    west, south, east, north = box
    sites = (
        airnow.loc[airnow.longitude.between(west, east) & airnow.latitude.between(south, north)]
        .drop_duplicates("station_id")[["station_id", "station_name", "latitude", "longitude"]]
    )
    start = int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp())
    end = start + 24 * 3600
    rows = []
    for site in sites.itertuples(index=False):
        query = urllib.parse.urlencode(
            {"lat": site.latitude, "lon": site.longitude, "start": start, "end": end, "appid": key}
        )
        payload = json.loads(_fetch(f"{OWM_HISTORY}?{query}"))
        for entry in payload.get("list", []):
            rows.append(
                {
                    "station_id": site.station_id,
                    "station_name": site.station_name,
                    "latitude": site.latitude,
                    "longitude": site.longitude,
                    "time_utc": datetime.fromtimestamp(entry["dt"], UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "no2_ugm3": entry["components"].get("no2"),
                    "aqi": entry["main"].get("aqi"),
                }
            )
        print(f"  OpenWeatherMap history at {site.station_name}")
    frame = pd.DataFrame(rows)
    path = out / "owm_no2_history_nyc.csv"
    frame.to_csv(path, index=False)
    print(f"  wrote {path.relative_to(ROOT)}  ({len(frame):,} rows)")
    # The API key is deliberately not recorded anywhere.
    return {"owm_no2_history_nyc.csv": {"urls": [OWM_HISTORY], "rows": len(frame)}}


# ------------------------------------------------------------------------- main

SOURCES = ("geoscf", "gfs", "era5", "airnow", "cams", "owm")
PUBLIC = ("geoscf", "gfs", "era5", "airnow")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", type=date.fromisoformat, default=date(2026, 6, 15))
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX,
                        metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--only", nargs="+", choices=SOURCES, default=list(PUBLIC))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    out = args.output or ROOT / "data" / "context" / "no2-intercomparison" / args.date.isoformat()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}

    stagers = {
        "geoscf": lambda: stage_geoscf(args.date, DEFAULT_HOURS, args.bbox, out),
        "gfs": lambda: stage_gfs(args.date, DEFAULT_HOURS, args.bbox, out),
        "era5": lambda: stage_era5(args.date, DEFAULT_HOURS, args.bbox, out),
        "airnow": lambda: stage_airnow(args.date, args.bbox, out),
        "cams": lambda: stage_cams(args.date, DEFAULT_HOURS, args.bbox, out),
        "owm": lambda: stage_owm(args.date, out),
    }
    for name in args.only:
        print(f"[{name}]")
        entries = stagers[name]()
        for filename, details in entries.items():
            manifest["files"][filename] = {
                **details,
                "staged_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            }

    manifest.update(
        {
            "case": "TEMPO NO2 forecast intercomparison",
            "date": args.date.isoformat(),
            "bbox": list(args.bbox),
            "hours_utc": list(DEFAULT_HOURS),
            "tempo_store": "data/tempo/northeast.zarr",
            "caveats": [
                "AirNow values are preliminary and not regulatory AQS data.",
                "CAMS and GEOS-CF assimilate satellite observations; neither is fully independent of TEMPO.",
                "ERA5 is a reanalysis, not a forecast.",
                "DestinE Extremes DT output is retained for ~15 days and could not be staged for this date.",
            ],
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote provenance to {manifest_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
