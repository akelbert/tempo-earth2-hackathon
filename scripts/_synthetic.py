"""Synthetic TEMPO and forecast data with a known answer.

Shared by ``smoke_test.py`` and ``test_notebook.py``. The point of building the
observation *from* the wind is that the correct advection result is known
exactly, so the tests can assert on physics rather than on "it did not crash".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

EARTH_RADIUS_M = 6_371_000.0

# One synthetic scene: a few urban plumes over the northeast corridor, drifting
# with a constant wind. Roughly the geometry of Boston, New York, Philadelphia.
SOURCES = [(42.35, -71.06), (40.71, -74.01), (39.95, -75.16)]


def displacement_degrees(u_ms: float, v_ms: float, dt_hours: float, lat0: float):
    """The lat/lon displacement a constant wind produces over ``dt_hours``."""
    dt = dt_hours * 3600.0
    dlat = np.degrees(v_ms * dt / EARTH_RADIUS_M)
    dlon = np.degrees(u_ms * dt / (EARTH_RADIUS_M * np.cos(np.radians(lat0))))
    return dlat, dlon


def plume_field(lat, lon, centres, width_deg=0.35, amplitude=8e15):
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    field = np.zeros_like(la)
    for lat0, lon0 in centres:
        field += amplitude * np.exp(
            -(((la - lat0) ** 2 + (lo - lon0) ** 2) / (2 * width_deg**2))
        )
    return field


def build_tempo(
    path: str | Path,
    u_ms: float = 7.5,
    v_ms: float = -3.0,
    dt_hours: float = 1.0,
    nscans: int = 4,
    lat_range=(38.5, 43.5),
    lon_range=(-76.0, -70.0),
    resolution: float = 0.02,
    start: str = "2026-06-15T16:00:00",
) -> xr.Dataset:
    """Write a synthetic TEMPO-shaped Zarr store.

    Scan *n* is scan 0 with every plume translated by *n* times the
    displacement the wind implies. Advecting scan *n* by that wind must
    therefore reproduce scan *n+1*.
    """
    lat = np.arange(*lat_range, resolution)
    lon = np.arange(*lon_range, resolution)
    dlat, dlon = displacement_degrees(u_ms, v_ms, dt_hours, np.mean(lat_range))

    background = 1.5e15 * (1.0 + 0.1 * np.sin(np.radians(lon))[None, :])

    scans = []
    for n in range(nscans):
        centres = [(la + n * dlat, lo + n * dlon) for la, lo in SOURCES]
        scans.append(plume_field(lat, lon, centres) + background)
    no2 = np.stack(scans)

    t0 = np.datetime64(start, "ns")
    times = np.array(
        [t0 + np.timedelta64(int(n * dt_hours * 3600), "s") for n in range(nscans)]
    )

    ds = xr.Dataset(
        {
            "no2_trop": (("time", "lat", "lon"), no2),
            "no2_strat": (("time", "lat", "lon"), np.full_like(no2, 3e15)),
            "qa_flag": (("time", "lat", "lon"), np.zeros_like(no2, dtype=np.int8)),
            "cloud_fraction": (("time", "lat", "lon"), np.full_like(no2, 0.05)),
        },
        coords={"time": times, "lat": lat, "lon": lon},
        attrs={"title": "synthetic TEMPO-shaped test data", "synthetic": "yes"},
    )
    ds.to_zarr(str(path), mode="w", consolidated=True)
    return ds


def build_forecast(
    u_ms: float = 7.5,
    v_ms: float = -3.0,
    init: str = "2026-06-15T12:00:00",
    leads_hours=(0, 6, 12),
) -> xr.Dataset:
    """A globally uniform wind field shaped like ``forecast.to_xarray`` output."""
    lat = np.linspace(-90, 89.75, 720)
    lon = np.linspace(-180, 179.75, 1440)
    lead = np.array([np.timedelta64(h, "h") for h in leads_hours])
    time = np.array([np.datetime64(init, "ns")])
    shape = (len(time), len(lead), len(lat), len(lon))

    def const(value):
        return (("time", "lead_time", "lat", "lon"), np.full(shape, value, dtype="float32"))

    ds = xr.Dataset(
        {
            "u10m": const(u_ms),
            "v10m": const(v_ms),
            "u100m": const(u_ms * 1.2),
            "v100m": const(v_ms * 1.2),
            "u850": const(u_ms * 1.4),
            "v850": const(v_ms * 1.4),
            "t2m": const(295.0),
            "msl": const(101325.0),
        },
        coords={"time": time, "lead_time": lead, "lat": lat, "lon": lon},
        attrs={"synthetic": "yes"},
    )
    return ds.assign_coords(valid_time=ds["time"] + ds["lead_time"])


def write_precomputed(directory: str | Path, model: str = "FCN", **kwargs) -> Path:
    """Write a synthetic forecast where ``load_precomputed`` expects to find it."""
    ds = build_forecast(**kwargs)
    init = str(ds["time"].values[0])[:16].replace("-", "").replace(":", "")
    path = Path(directory) / f"{model.lower()}_{init}Z.zarr"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(str(path), mode="w", consolidated=True)
    return path
