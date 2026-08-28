"""Advect a TEMPO NO2 field with Earth-2 winds, and score the result.

This is the piece that makes the introductory notebook a *joint* TEMPO/Earth-2
workflow rather than two pictures side by side. The idea is deliberately simple
so it can be explained in one slide:

    NO2 near the surface is, over an hour or two, mostly transported rather than
    created or destroyed. So take the NO2 column TEMPO measured at scan N, push
    every pixel downwind using the wind Earth-2 forecast for that hour, and
    compare the result against what TEMPO actually measured at scan N+1.

The comparison is scored against persistence (do nothing). A skill score above
zero means the forecast winds carried real information about where the NO2 went.

Caveats worth stating to attendees, because they are the interesting part:

* A column is advected with a single wind level, but the column is deep. Which
  level to use is a genuine research choice, not a detail.
* Emission, chemistry, deposition and boundary-layer mixing are all ignored.
* Cloud-screened and low-quality pixels are missing data, and missing data moves
  around between scans, which affects the score independently of the physics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

EARTH_RADIUS_M = 6_371_000.0


@dataclass
class AdvectionSkill:
    """Scores for one advection experiment."""

    rmse_advected: float
    rmse_persistence: float
    skill_score: float
    n_valid: int
    dt_hours: float
    wind_level: str

    def __str__(self) -> str:
        verdict = "beats" if self.skill_score > 0 else "does not beat"
        return (
            f"{self.wind_level} advection over {self.dt_hours:.1f} h {verdict} "
            f"persistence: RMSE {self.rmse_advected:.3g} vs {self.rmse_persistence:.3g} "
            f"molecules/cm^2 (skill {self.skill_score:+.3f}, n={self.n_valid:,})"
        )


def _fractional_indices(
    lat: np.ndarray, lon: np.ndarray, u: np.ndarray, v: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Backward departure points, expressed as fractional grid indices."""
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

    dlat_deg = np.degrees(v * dt / EARTH_RADIUS_M)
    coslat = np.clip(np.cos(np.radians(lat2d)), 1e-6, None)
    dlon_deg = np.degrees(u * dt / (EARTH_RADIUS_M * coslat))

    lat_dep = lat2d - dlat_deg
    lon_dep = lon2d - dlon_deg

    dlat = float(lat[1] - lat[0])
    dlon = float(lon[1] - lon[0])
    return (lat_dep - lat[0]) / dlat, (lon_dep - lon[0]) / dlon


def advect(
    field: xr.DataArray,
    u: xr.DataArray,
    v: xr.DataArray,
    dt_hours: float,
    substeps: int = 4,
    min_weight: float = 0.5,
) -> xr.DataArray:
    """Semi-Lagrangian advection of ``field`` by winds ``u``/``v``.

    ``field``, ``u`` and ``v`` must already share the same ``lat``/``lon`` grid
    (see :func:`tempo_earth2.forecast.regrid_to`). Winds are in m/s and held
    fixed over the interval, which is the main approximation.

    Missing data (cloud-screened or quality-screened pixels) is handled by
    advecting a companion weight field of ones and renormalizing, so NaNs
    spread only as far as they physically should rather than contaminating
    every downwind pixel.
    """
    from scipy.ndimage import map_coordinates

    lat = field["lat"].values.astype(float)
    lon = field["lon"].values.astype(float)
    values = field.values.astype(float)
    uu = np.broadcast_to(u.values.astype(float), values.shape).copy()
    vv = np.broadcast_to(v.values.astype(float), values.shape).copy()

    valid = np.isfinite(values)
    current = np.where(valid, values, 0.0)
    weight = valid.astype(float)

    dt = dt_hours * 3600.0 / max(1, substeps)
    for _ in range(max(1, substeps)):
        rows, cols = _fractional_indices(lat, lon, uu, vv, dt)
        coords = np.stack([rows, cols])
        # mode="constant", cval=0 on both arrays drives the weight to zero
        # outside the domain, so the upwind boundary strip comes back as NaN.
        # That is the honest answer: we do not know what blew in from outside
        # the region. Clamping to the edge value instead would fabricate inflow
        # and then quietly score it.
        current = map_coordinates(current, coords, order=1, mode="constant", cval=0.0)
        weight = map_coordinates(weight, coords, order=1, mode="constant", cval=0.0)

    out = np.divide(current, weight, out=np.full_like(current, np.nan), where=weight > 0)
    out[weight < min_weight] = np.nan

    result = xr.DataArray(
        out,
        coords={"lat": field["lat"], "lon": field["lon"]},
        dims=("lat", "lon"),
        name=f"{field.name}_advected",
        attrs={
            **field.attrs,
            "advection_dt_hours": dt_hours,
            "advection_substeps": substeps,
            "advection_note": "semi-Lagrangian, wind held constant, no chemistry",
        },
    )
    return result


def score(
    advected: xr.DataArray,
    observed: xr.DataArray,
    persistence: xr.DataArray,
    dt_hours: float,
    wind_level: str = "u10m/v10m",
) -> AdvectionSkill:
    """Compare advected and persisted fields against a later observation.

    Only pixels where all three fields are valid are scored, so the advected and
    persistence RMSEs are computed on identical samples and the comparison is
    fair.
    """
    a = advected.values.astype(float)
    o = observed.values.astype(float)
    p = persistence.values.astype(float)

    mask = np.isfinite(a) & np.isfinite(o) & np.isfinite(p)
    n = int(mask.sum())
    if n == 0:
        raise ValueError(
            "No pixels are valid in all three fields. Cloud screening probably "
            "removed the overlap - try a different scan pair or relax "
            "max_cloud_fraction."
        )

    rmse_a = float(np.sqrt(np.mean((a[mask] - o[mask]) ** 2)))
    rmse_p = float(np.sqrt(np.mean((p[mask] - o[mask]) ** 2)))
    skill = 1.0 - rmse_a / rmse_p if rmse_p > 0 else float("nan")

    return AdvectionSkill(
        rmse_advected=rmse_a,
        rmse_persistence=rmse_p,
        skill_score=skill,
        n_valid=n,
        dt_hours=dt_hours,
        wind_level=wind_level,
    )


def experiment(
    no2_t0: xr.DataArray,
    no2_t1: xr.DataArray,
    u: xr.DataArray,
    v: xr.DataArray,
    dt_hours: float,
    wind_level: str = "u10m/v10m",
    substeps: int = 4,
) -> tuple[xr.DataArray, AdvectionSkill]:
    """Run one advection experiment end to end and return field plus score."""
    advected = advect(no2_t0, u, v, dt_hours=dt_hours, substeps=substeps)
    result = score(advected, no2_t1, no2_t0, dt_hours=dt_hours, wind_level=wind_level)
    return advected, result


def wind_speed(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Scalar wind speed, named and unit-tagged for plotting."""
    speed = np.sqrt(u**2 + v**2)
    speed.name = "wind_speed"
    speed.attrs["units"] = "m s-1"
    return speed


def ventilation_relationship(
    no2: xr.DataArray, speed: xr.DataArray, bins: int = 12
) -> xr.Dataset:
    """Median NO2 column binned by wind speed.

    The expected signature of ventilation is a decreasing curve: faster winds
    disperse the same emissions over a larger area, so the column above any
    given point is lower. It is a quick, honest sanity check that the model
    winds and the TEMPO retrieval are describing the same atmosphere.
    """
    n = np.asarray(no2.values, dtype=float).ravel()
    s = np.asarray(speed.values, dtype=float).ravel()
    mask = np.isfinite(n) & np.isfinite(s)
    n, s = n[mask], s[mask]
    if n.size == 0:
        raise ValueError("No overlapping valid NO2 and wind pixels.")

    edges = np.linspace(np.nanpercentile(s, 1), np.nanpercentile(s, 99), bins + 1)
    idx = np.clip(np.digitize(s, edges) - 1, 0, bins - 1)

    centres = 0.5 * (edges[:-1] + edges[1:])
    median = np.full(bins, np.nan)
    count = np.zeros(bins, dtype=int)
    for b in range(bins):
        sel = idx == b
        count[b] = int(sel.sum())
        if count[b]:
            median[b] = float(np.median(n[sel]))

    return xr.Dataset(
        {
            "median_no2": ("wind_speed", median),
            "count": ("wind_speed", count),
        },
        coords={"wind_speed": centres},
    )
