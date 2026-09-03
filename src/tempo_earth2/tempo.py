"""Loading and subsetting NASA TEMPO gridded NO2 (Level 3).

Two access paths are supported, and the notebooks prefer them in this order:

1. **Staged copy** (``TEMPO_DATA_URI``) - a Zarr store or directory of netCDF
   files prepared by ``scripts/stage_tempo.py``. This is what attendees use on
   event day: it is regional, variable-subset, and lives next to the cluster.
2. **Direct from NASA Earthdata** via ``earthaccess``. This needs an Earthdata
   login and pulls ~850 MB per full-disk granule, so it is a preparation-time
   path, not an event-day path.

The normalized dataset returned by every loader here uses the coordinate names
``time``, ``lat``, ``lon`` (longitude in -180..180) regardless of which path
produced it, so downstream code never has to branch.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr

#: Canonical variable name for the tropospheric NO2 vertical column.
NO2_TROPOSPHERIC = "no2_trop"

#: Variables kept when staging or loading. Maps TEMPO name -> workshop name.
TEMPO_VARIABLES = {
    "vertical_column_troposphere": NO2_TROPOSPHERIC,
    "vertical_column_stratosphere": "no2_strat",
    "vertical_column_total": "no2_total",
    "main_data_quality_flag": "qa_flag",
}

_GRANULE_TIME = re.compile(r"_(\d{8}T\d{6}Z)_S(\d+)")


@dataclass(frozen=True)
class Region:
    """A named lon/lat bounding box, in degrees, longitude in -180..180."""

    name: str
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(west, south, east, north), the ordering ``earthaccess`` expects."""
        return (self.lon_min, self.lat_min, self.lon_max, self.lat_max)


#: Regions sized to be interesting for NO2 transport and small enough that a
#: forecast/observation comparison runs in seconds rather than minutes.
REGIONS: dict[str, Region] = {
    # The I-95 corridor: dense, well-separated urban sources, strong and
    # frequently along-corridor flow. The workshop default.
    "northeast": Region("northeast", -80.0, 36.0, -66.0, 45.0),
    "chicago": Region("chicago", -92.0, 39.0, -83.0, 45.0),
    "losangeles": Region("losangeles", -121.0, 32.0, -114.0, 37.0),
    "texas": Region("texas", -100.0, 27.0, -92.0, 34.0),
    "conus": Region("conus", -125.0, 24.0, -66.0, 50.0),
}


def granule_time(name: str) -> datetime | None:
    """Parse the scan start time out of a TEMPO granule filename."""
    match = _GRANULE_TIME.search(str(name))
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(
        tzinfo=UTC
    )


def _normalize(ds: xr.Dataset) -> xr.Dataset:
    """Rename coordinates to time/lat/lon and put longitude in -180..180."""
    renames = {}
    for candidate, target in (
        ("latitude", "lat"),
        ("longitude", "lon"),
        ("Latitude", "lat"),
        ("Longitude", "lon"),
    ):
        if candidate in ds.dims or candidate in ds.coords or candidate in ds.variables:
            renames[candidate] = target
    if renames:
        ds = ds.rename(renames)

    if "lon" in ds.coords:
        lon = ds["lon"].values
        if np.nanmax(lon) > 180.0:
            ds = ds.assign_coords(lon=(((lon + 180.0) % 360.0) - 180.0)).sortby("lon")
    if "lat" in ds.coords and ds["lat"].size > 1 and ds["lat"][0] > ds["lat"][-1]:
        ds = ds.sortby("lat")
    return ds


def open_tempo_granule(path: str | Path) -> xr.Dataset:
    """Open one TEMPO L3 netCDF granule as a flat, normalized dataset.

    TEMPO L3 files are hierarchical netCDF4: ``latitude``/``longitude``/``time``
    live in the root group while the retrieved columns live in ``/product``.
    xarray cannot open several groups at once, so we open both and merge.
    """
    path = str(path)
    root = xr.open_dataset(path, engine="h5netcdf", decode_times=True)

    try:
        product = xr.open_dataset(
            path, group="product", engine="h5netcdf", decode_times=False
        )
    except (OSError, KeyError, ValueError):
        # Already-flattened file (for example a staged subset saved as netCDF).
        return _normalize(root)

    keep = {src: dst for src, dst in TEMPO_VARIABLES.items() if src in product}
    product = product[list(keep)].rename(keep)

    coords = {
        name: root[name] for name in ("latitude", "longitude", "time") if name in root
    }
    merged = product.assign_coords(coords)
    merged.attrs.update(root.attrs)
    return _normalize(merged)


def open_tempo_source(uri: str) -> xr.Dataset:
    """Open the staged workshop TEMPO copy.

    ``uri`` may be a Zarr store (local path or ``gs://``/``s3://``), a directory
    of netCDF granules, or a single netCDF file. Zarr is the event-day format;
    the others exist so the same notebook works on a laptop.
    """
    uri = uri.rstrip("/")

    if uri.endswith(".zarr"):
        return _normalize(xr.open_zarr(uri, consolidated=True, decode_times=True))

    if uri.startswith(("gs://", "s3://", "http://", "https://")):
        # Remote and not obviously Zarr: assume Zarr anyway, that is the only
        # remote layout the staging script produces.
        return _normalize(xr.open_zarr(uri, consolidated=True, decode_times=True))

    path = Path(uri)
    if path.is_dir():
        zarrs = sorted(path.glob("*.zarr"))
        if zarrs:
            return _normalize(
                xr.open_zarr(str(zarrs[0]), consolidated=True, decode_times=True)
            )
        granules = sorted(path.glob("**/*.nc"))
        if not granules:
            raise FileNotFoundError(
                f"No .zarr store and no .nc granules found under {path}. "
                "Run scripts/stage_tempo.py, or set TEMPO_DATA_URI."
            )
        return load_tempo_series(granules)

    if path.is_file():
        return open_tempo_granule(path)

    raise FileNotFoundError(
        f"TEMPO_DATA_URI={uri!r} does not exist. Run scripts/stage_tempo.py to "
        "stage a regional subset, or point TEMPO_DATA_URI at an existing copy."
    )


def load_tempo_series(granules: Iterable[str | Path]) -> xr.Dataset:
    """Concatenate several TEMPO granules along ``time``, sorted by scan time."""
    granules = sorted(
        granules,
        key=lambda g: (granule_time(g) or datetime.min.replace(tzinfo=UTC), str(g)),
    )
    datasets = []
    for granule in granules:
        ds = open_tempo_granule(granule)
        if "time" not in ds.dims:
            stamp = granule_time(granule)
            if stamp is None:
                raise ValueError(f"Cannot determine scan time for {granule}")
            ds = ds.expand_dims(time=[np.datetime64(stamp.replace(tzinfo=None), "ns")])
        datasets.append(ds)
    if not datasets:
        raise ValueError("No TEMPO granules to load")
    return xr.concat(datasets, dim="time", combine_attrs="override")


def subset(
    ds: xr.Dataset,
    region: Region | str | None = None,
    variables: Sequence[str] | None = None,
) -> xr.Dataset:
    """Restrict a TEMPO dataset to a region and variable list."""
    if isinstance(region, str):
        region = REGIONS[region]
    if region is not None:
        ds = ds.sel(
            lat=slice(region.lat_min, region.lat_max),
            lon=slice(region.lon_min, region.lon_max),
        )
    if variables is not None:
        ds = ds[[v for v in variables if v in ds]]
    return ds


def apply_quality_mask(
    ds: xr.Dataset,
    variable: str = NO2_TROPOSPHERIC,
    max_flag: int = 0,
    max_cloud_fraction: float | None = 0.2,
) -> xr.DataArray:
    """Return ``variable`` with low-quality retrievals set to NaN.

    ``main_data_quality_flag`` is 0 for normal (good) retrievals; the workshop
    default keeps only those. Cloud screening is applied when the staged copy
    carries an effective cloud fraction.
    """
    field = ds[variable]
    if "qa_flag" in ds:
        field = field.where(ds["qa_flag"] <= max_flag)
    if max_cloud_fraction is not None and "cloud_fraction" in ds:
        field = field.where(ds["cloud_fraction"] <= max_cloud_fraction)
    return field


def load_tempo_scan(
    uri: str | None = None,
    time: str | np.datetime64 | None = None,
    region: Region | str | None = "northeast",
) -> xr.Dataset:
    """Load a single TEMPO scan, regionally subset - the notebook entry point.

    Parameters
    ----------
    uri
        Staged TEMPO location. Defaults to ``$TEMPO_DATA_URI``.
    time
        Requested scan time. The nearest available scan is returned. ``None``
        selects the first scan in the staged copy.
    region
        Named region or explicit :class:`Region`. ``None`` keeps the full grid.
    """
    uri = uri or os.environ.get("TEMPO_DATA_URI")
    if not uri:
        raise ValueError(
            "No TEMPO location given. Set TEMPO_DATA_URI or pass uri=... ."
        )
    ds = subset(open_tempo_source(uri), region=region)
    if "time" not in ds.dims:
        return ds
    if time is None:
        return ds.isel(time=0)
    return ds.sel(time=np.datetime64(time), method="nearest")


def search_earthdata(
    region: Region | str,
    start: str,
    end: str,
    short_name: str = "TEMPO_NO2_L3",
    version: str = "V04",
) -> list:
    """Search NASA Earthdata for TEMPO L3 granules (preparation-time path).

    Requires ``earthaccess`` and a NASA Earthdata login. Call
    ``earthaccess.login(persist=True)`` first, or set ``EARTHDATA_USERNAME``
    and ``EARTHDATA_PASSWORD``.
    """
    import earthaccess

    if isinstance(region, str):
        region = REGIONS[region]
    return earthaccess.search_data(
        short_name=short_name,
        version=version,
        temporal=(start, end),
        bounding_box=region.bbox,
    )
