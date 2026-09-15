# %% [markdown]
# # TEMPO NO₂ against weather and chemical forecasts
#
# **Northeast US and New York City, 15 June 2026.** Run notebook 04 first: this
# notebook reuses its transport experiment and assumes you have seen it.
#
# ---
#
# ## The question
#
# TEMPO measured the NO₂ column over the Northeast six times on 15 June 2026.
# How well did the forecasts available that day anticipate what it saw? There
# are two different kinds of forecast to ask:
#
# 1. **Weather forecasts move NO₂ around.** Earth-2's FourCastNet (FCN), NOAA's
#    GFS and the ERA5 reanalysis all give a wind. None of them carries NO₂, so
#    we test them the way notebook 04 does: advect one TEMPO scan with the wind
#    and score it against the next scan.
# 2. **Chemical forecasts predict NO₂ itself.** CAMS (ECMWF) and GEOS-CF (NASA)
#    run chemistry and emissions. Those we compare to TEMPO directly, column
#    against column, and to ground monitors, surface against surface.
#
# ## Sources in this notebook
#
# | Source | What it gives | Grid | Kind |
# | --- | --- | --- | --- |
# | TEMPO L3 V04 | tropospheric and stratospheric NO₂ column | ~2 km, hourly scans | observation |
# | Earth-2 FCN | 10 m / 100 m / 850 hPa wind | 0.25°, 6-hourly | AI weather forecast, run here from GFS 12Z |
# | NOAA GFS | same winds | 0.25°, hourly | physics weather forecast, 12Z cycle |
# | ERA5 (ERA5T) | same winds | 0.25°, hourly | reanalysis — has seen the future |
# | CAMS global | total NO₂ column, lowest-level NO₂ | 0.4°, hourly / 3-hourly | chemical forecast, 00Z |
# | GEOS-CF v2 | surface NO₂ forecast; NO₂ columns from the analysis | 0.25°, hourly | chemical forecast / analysis |
# | EPA AirNow | surface NO₂ at monitors | points, hourly | observation (preliminary) |
# | OpenWeatherMap | surface NO₂ at the monitors | points, hourly | commercial air-quality model history |
#
# Everything except FCN is a small staged extract in
# `data/context/no2-intercomparison/2026-06-15/`, written by
# `scripts/stage_no2_intercomparison.py`; its `manifest.json` records every
# source URL. FCN runs live on the Hub GPU.
#
# **What is not here, and why.** DestinE's Extremes Digital Twin keeps its
# global forecasts for only about 15 days, so 15 June could not be retrieved;
# §3.6 shows where it slots in for a recent date. GEOS-CF's *forecast* columns
# are purged from the public portal after about two weeks, so the column
# comparison uses the GEOS-CF *analysis* instead.
#
# ## Running time
#
# About two minutes on the Hub, most of it the FCN model load. Without a GPU
# the notebook substitutes the synthetic reference forecast and says so loudly.

# %% [markdown]
# ---
# ## 0. Settings

# %%
# --- What to look at ----------------------------------------------------------
CASE_DATE = "2026-06-15"
REGION = "northeast"
CITY_NAME = "New York City"
CITY_BOX = (-74.30, 40.45, -73.65, 41.00)  # west, south, east, north
SCAN_INDEX = 1            # which scan is t0 for the maps (0 = first)
SCAN_GAP = 1              # how many scans ahead t1 is

# --- Data quality -------------------------------------------------------------
MAX_QA_FLAG = 0           # 0 keeps only normal-quality retrievals
MAX_CLOUD_FRACTION = 0.2  # raise towards 1.0 to keep more (cloudier) pixels
MIN_COVERAGE = 0.5        # share of TEMPO pixels that must be valid in a model cell

# --- Weather forecasts --------------------------------------------------------
MODEL_NAME = "FCN"        # see tempo_earth2.forecast.SUPPORTED_PROGNOSTICS
WIND_LEVEL = "10m"        # "10m" | "100m" | "850"
FCN_MODE = "auto"         # "live" | "reference" | "auto" (live when a GPU is visible)

# %%
import json
import os
import warnings
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from tempo_earth2 import advect, context, forecast, plots
from tempo_earth2.config import WorkshopConfig
from tempo_earth2.tempo import REGIONS, apply_quality_mask, open_tempo_source, subset

warnings.filterwarnings("ignore", category=RuntimeWarning)

config = WorkshopConfig.from_env()
config.ensure_dirs()
region = REGIONS[REGION]

CASE_RELATIVE = Path("data") / "context" / "no2-intercomparison" / CASE_DATE


def find_case_dir() -> Path:
    """Locate the staged case: $NO2_CASE_DIR, this repository, then the context root."""
    candidates = []
    if os.environ.get("NO2_CASE_DIR"):
        candidates.append(Path(os.environ["NO2_CASE_DIR"]))
    for base in (Path.cwd(), *Path.cwd().parents):
        candidates.append(base / CASE_RELATIVE)
    candidates.append(Path(context.data_uri(f"no2-intercomparison/{CASE_DATE}")))
    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return candidate
    raise FileNotFoundError(
        "Staged NO2 intercomparison case not found. Open this notebook from a clone "
        "of the repository, set NO2_CASE_DIR, or run "
        "scripts/stage_no2_intercomparison.py."
    )


CASE_DIR = find_case_dir()
REPO_DATA = CASE_DIR.parents[2]  # .../data
manifest = json.loads((CASE_DIR / "manifest.json").read_text())


def open_case(name: str) -> xr.Dataset | None:
    """Open one staged gridded file, or return None if it has not been staged."""
    path = CASE_DIR / name
    if not path.is_file():
        print(f"  not staged: {name}")
        return None
    return xr.open_dataset(path).load()


print(f"Region:        {region.name}  {region.bbox}")
print(f"Staged case:   {CASE_DIR}")
print(f"Staged files:  {', '.join(sorted(manifest['files']))}")
print(f"GPU:           {'yes' if forecast.cuda_available() else 'no'}")

# %% [markdown]
# ---
# ## 1. The observation
#
# Same screening as notebook 04: `main_data_quality_flag == 0` and effective
# cloud fraction ≤ 0.2. Screened pixels are *missing*, not clean.
#
# The staged TEMPO copy is checked against `CASE_DATE`. If the configured
# workshop store holds a different day, the notebook falls back to the copy in
# this repository rather than silently comparing different days.

# %%
def open_case_tempo() -> tuple[xr.Dataset, str]:
    candidates = [config.tempo_uri, str(REPO_DATA / "tempo" / "northeast.zarr")]
    for uri in candidates:
        try:
            ds = subset(open_tempo_source(uri), region=region)
        except Exception as exc:  # noqa: BLE001 - try the next copy
            print(f"  skipped {uri}: {type(exc).__name__}")
            continue
        if str(ds["time"].values[0])[:10] == CASE_DATE:
            return ds, uri
        print(f"  skipped {uri}: holds {str(ds['time'].values[0])[:10]}, not {CASE_DATE}")
    raise FileNotFoundError(f"No TEMPO copy for {CASE_DATE}")


tempo, tempo_uri = open_case_tempo()
scan_times = tempo["time"].values
print(f"TEMPO source: {tempo_uri}")
print(f"{len(scan_times)} scans:")
for i, t in enumerate(scan_times):
    marker = "  <- t0" if i == SCAN_INDEX else ("  <- t1" if i == SCAN_INDEX + SCAN_GAP else "")
    print(f"  [{i}] {np.datetime_as_string(t, unit='m')}Z{marker}")


def screened(i: int, variable: str = "no2_trop") -> xr.DataArray:
    return apply_quality_mask(
        tempo.isel(time=i),
        variable=variable,
        max_flag=MAX_QA_FLAG,
        max_cloud_fraction=MAX_CLOUD_FRACTION,
    )


def as_utc(t) -> datetime:
    return pd.Timestamp(t).floor("s").tz_localize("UTC").to_pydatetime()


# A TEMPO L3 scan takes about an hour to sweep the Northeast; its time stamp is
# the scan start. Models are compared at the middle of the scan.
SCAN_DURATION = pd.Timedelta(minutes=30)
scan_mid = [(pd.Timestamp(t) + SCAN_DURATION).floor("min") for t in scan_times]

t0, t1 = scan_times[SCAN_INDEX], scan_times[SCAN_INDEX + SCAN_GAP]
no2_t0, no2_t1 = screened(SCAN_INDEX), screened(SCAN_INDEX + SCAN_GAP)
dt_hours = float((t1 - t0) / np.timedelta64(1, "h"))
print(f"\nValid pixels at t0: {float(np.isfinite(no2_t0.values).mean()):.1%}")

# %%
ax, _ = plots.plot_no2(
    no2_t0, title=f"TEMPO tropospheric NO$_2$  {np.datetime_as_string(t0, unit='m')}Z"
)
west, south, east, north = CITY_BOX
ax.plot([west, east, east, west, west], [south, south, north, north, south],
        color="tab:blue", linewidth=1.2, transform=getattr(ax, "projection", None) or ax.transData)
ax.set_title(ax.get_title() + f"   (box: {CITY_NAME})", fontsize=10);

# %% [markdown]
# ---
# ## 2. Weather forecasts: whose wind moves the NO₂ best?
#
# ### 2.1 Earth-2 FCN
#
# FCN steps every 6 hours and is initialized from GFS, so we snap *down* to the
# 12Z cycle before the first scan. `FCN_MODE = "auto"` runs the model when a GPU
# is visible. Without one it loads the reference forecast, which in this
# repository is a **synthetic** teaching fixture — fine for checking that the
# code runs, meaningless as a score.

# %%
init_time = forecast.nearest_init_time(as_utc(scan_times[0]))
nsteps = forecast.steps_to_cover(init_time, as_utc(scan_times[-1]))
use_live = FCN_MODE == "live" or (FCN_MODE == "auto" and forecast.cuda_available())

if use_live:
    fcn_run = forecast.run_forecast(
        init_time.replace(tzinfo=None),
        nsteps=nsteps,
        model_name=MODEL_NAME,
        variables=forecast.TRANSPORT_VARIABLES,
        store_path=config.outputs / f"{MODEL_NAME.lower()}_{init_time:%Y%m%dT%H%M}Z.zarr",
    )
    fx = fcn_run.dataset
    fcn_summary = fcn_run.summary()
else:
    path = forecast.precomputed_forecast_path(config, MODEL_NAME, init_time)
    if not path.exists():
        path = REPO_DATA / "precomputed" / path.name
    fx = forecast.load_precomputed(path)
    fcn_summary = {"source": "reference", "store": str(path), "synthetic": fx.attrs.get("synthetic")}

FCN_SYNTHETIC = str(fx.attrs.get("synthetic", "")).lower() in {"yes", "true", "1"}
fx = fx.sel(lat=slice(south - 12, north + 12)).sortby("lat")
print(f"FCN init {init_time:%Y-%m-%d %H:%M}Z, {nsteps} steps, source: {fcn_summary}")
if FCN_SYNTHETIC:
    print("\n  *** FCN winds below are SYNTHETIC. Run on a GPU for a real FCN score. ***")

# %% [markdown]
# ### 2.2 GFS and ERA5
#
# GFS is the operational forecast FCN was initialized from, with hourly output.
# ERA5 is the reference: a reanalysis, so it is not a fair *forecast* competitor
# — it shows how much skill is available from a wind that is as right as we can
# make it at 0.25°.

# %%
gfs = open_case("gfs_fcst_winds.nc")
era5 = open_case("era5_winds.nc")

LEVELS = {"10m": ("u10m", "v10m"), "100m": ("u100m", "v100m"), "850": ("u850", "v850")}


def wind_at(source: str, valid: pd.Timestamp, level: str):
    """(u, v, lead/offset note) for one wind source at a valid time, native grid."""
    un, vn = LEVELS[level]
    if source == MODEL_NAME:
        picked = forecast.select_valid_time(fx, valid.floor("s").to_pydatetime())
        note = f"lead {picked.attrs['lead_hours']:.0f} h"
        return picked[un], picked[vn], note
    ds = {"GFS": gfs, "ERA5": era5}[source]
    picked = ds.sel(time=valid, method="nearest")
    offset = abs(pd.Timestamp(picked["time"].values) - valid) / pd.Timedelta(minutes=1)
    return picked[un], picked[vn], f"±{offset:.0f} min"


WIND_SOURCES = [MODEL_NAME] + [name for name, ds in (("GFS", gfs), ("ERA5", era5)) if ds is not None]
midpoint = pd.Timestamp(t0) + (pd.Timestamp(t1) - pd.Timestamp(t0)) / 2

winds = {}
for name in WIND_SOURCES:
    u_native, v_native, note = wind_at(name, midpoint, WIND_LEVEL)
    grid = forecast.regrid_to(
        xr.Dataset({"u": u_native, "v": v_native}), no2_t0["lat"].values, no2_t0["lon"].values
    )
    winds[name] = (grid["u"], grid["v"])
    print(f"{name:>5} {WIND_LEVEL} wind at {midpoint:%H:%M}Z ({note}): "
          f"mean speed {float(advect.wind_speed(grid['u'], grid['v']).mean()):.1f} m/s")

# %%
fig, axes = plt.subplots(1, len(WIND_SOURCES), figsize=(6 * len(WIND_SOURCES), 5),
                         squeeze=False, constrained_layout=True)
for ax, name in zip(axes[0], WIND_SOURCES):
    u, v = winds[name]
    every = max(1, no2_t0["lon"].size // 25)
    ax.pcolormesh(no2_t0.lon, no2_t0.lat, no2_t0 / plots.NO2_SCALE, cmap="magma_r",
                  vmin=0, vmax=np.nanpercentile(no2_t0 / plots.NO2_SCALE, 98), shading="auto")
    ax.quiver(u.lon[::every], u.lat[::every], u.values[::every, ::every],
              v.values[::every, ::every], color="tab:cyan", scale=350, width=0.003)
    label = f"{name}{' (SYNTHETIC)' if name == MODEL_NAME and FCN_SYNTHETIC else ''}"
    ax.set_title(f"TEMPO NO$_2$ with {label} {WIND_LEVEL} wind", fontsize=10)
    ax.set_xlabel("longitude")
axes[0][0].set_ylabel("latitude");

# %% [markdown]
# **Look before scoring.** Do the three winds agree on direction along the I-95
# corridor? Where they disagree is where the scores below can separate.
#
# ### 2.3 Ventilation check, for each wind
#
# Median NO₂ should fall as wind speed rises. A wind that fails this is not
# describing the same atmosphere as TEMPO.

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
for name in WIND_SOURCES:
    u, v = winds[name]
    rel = advect.ventilation_relationship(no2_t0, advect.wind_speed(u, v), bins=10)
    mask = np.isfinite(no2_t0.values)
    r = np.corrcoef(no2_t0.values[mask], advect.wind_speed(u, v).values[mask])[0, 1]
    ax.plot(rel.wind_speed, rel.median_no2 / plots.NO2_SCALE, marker="o", label=f"{name} (r = {r:+.2f})")
ax.set_xlabel(f"{WIND_LEVEL} wind speed (m s$^{{-1}}$)")
ax.set_ylabel(plots.NO2_LABEL)
ax.set_title("Ventilation: median TEMPO NO$_2$ against wind speed")
ax.grid(alpha=0.3)
ax.legend();

# %% [markdown]
# ### 2.4 Advect and score every scan pair
#
# One scan pair is an anecdote, so the experiment runs over all five
# consecutive pairs, at every wind level, for every source.
#
# ```
# skill = 1 - RMSE(advected) / RMSE(persistence)
# ```

# %%
records = []
for i in range(len(scan_times) - 1):
    a, b = screened(i), screened(i + 1)
    gap = float((scan_times[i + 1] - scan_times[i]) / np.timedelta64(1, "h"))
    mid = pd.Timestamp(scan_times[i]) + (pd.Timestamp(scan_times[i + 1]) - pd.Timestamp(scan_times[i])) / 2
    for name in WIND_SOURCES:
        for level in LEVELS:
            try:
                u_native, v_native, note = wind_at(name, mid, level)
            except KeyError:
                continue
            grid = forecast.regrid_to(xr.Dataset({"u": u_native, "v": v_native}), a.lat.values, a.lon.values)
            try:
                _, s = advect.experiment(a, b, grid["u"], grid["v"], dt_hours=gap, wind_level=f"{name} {level}")
            except ValueError as exc:
                print(f"  pair {i}->{i + 1} {name} {level}: {exc}")
                continue
            records.append({
                "pair": f"{pd.Timestamp(scan_times[i]):%H:%M}->{pd.Timestamp(scan_times[i + 1]):%H:%M}Z",
                "source": name, "level": level, "skill": s.skill_score,
                "rmse_advected": s.rmse_advected, "rmse_persistence": s.rmse_persistence,
                "n_valid": s.n_valid, "wind_time": note,
            })

transport = pd.DataFrame(records)
transport_summary = transport.pivot_table(index="source", columns="level", values="skill", aggfunc="mean")
print("Mean skill over all scan pairs (positive beats persistence):")
transport_summary.round(3)

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
for name in WIND_SOURCES:
    rows = transport.query("source == @name and level == @WIND_LEVEL")
    ax.plot(rows["pair"], rows["skill"], marker="o", label=name)
ax.axhline(0, color="k", linewidth=0.8)
ax.set_ylabel("skill vs persistence")
ax.set_title(f"Advection skill by scan pair, {WIND_LEVEL} wind")
ax.grid(alpha=0.3)
ax.legend();

# %%
best = transport.query("level == @WIND_LEVEL").groupby("source")["skill"].mean().idxmax()
advected, skill = advect.experiment(no2_t0, no2_t1, *winds[best], dt_hours=dt_hours,
                                    wind_level=f"{best} {WIND_LEVEL}")
plots.plot_advection_triptych(no2_t0, advected, no2_t1, skill=skill);

# %% [markdown]
# ### 2.5 Reading the transport scores
#
# * **Time resolution.** GFS and ERA5 are hourly; FCN is 6-hourly, so its wind
#   can be up to three hours from the interval it is applied to. A better FCN
#   score in spite of that would be notable.
# * **ERA5 is not a forecast.** It is the ceiling for a 0.25° wind, not a rival.
# * **FCN starts from GFS 12Z.** Differences between FCN and GFS at a 2–7 h lead
#   are mostly model error growth, not different initial conditions.
# * All the caveats of notebook 04 still apply: one level for a deep column, no
#   chemistry, and a 25 km wind against 2 km structure.
#
# ### 2.6 Where DestinE fits
#
# The Extremes DT produces a 4.4 km global forecast every day, but its data are
# only retrievable for about 15 days. For a recent date, and with DestinE
# Digital Twin access, the winds come through `earthkit-data` and Polytope:
#
# ```python
# import earthkit.data
# request = {
#     "class": "d1", "dataset": "extremes-dt", "expver": "0001", "stream": "oper",
#     "type": "fc", "levtype": "sfc", "param": "165/166",   # 10 m u/v
#     "date": "YYYYMMDD", "time": "0000", "step": "14/15/16/17/18/19",
# }
# dt = earthkit.data.from_source("polytope", "destination-earth", request, stream=False)
# ```
#
# Regrid it to 0.25° (or to the TEMPO grid directly — it is much closer to it),
# crop to the region, add it to `WIND_SOURCES` with the same `u10m`/`v10m`
# names, and every table above extends by one row.

# %% [markdown]
# ---
# ## 3. Chemical forecasts against the TEMPO column
#
# CAMS and GEOS-CF carry NO₂, so they can be compared to TEMPO directly — but
# only on *their* grid. Interpolating a 0.25° model onto TEMPO's 2 km pixels
# would invent structure the model does not have, so we go the other way:
# average the valid TEMPO pixels inside each model cell, and keep the cell only
# if at least `MIN_COVERAGE` of its pixels survived cloud and quality screening.
#
# Two like-for-like pairs:
#
# * **tropospheric column**: TEMPO `no2_trop` against GEOS-CF `TropCol_NO2`;
# * **total column**: TEMPO `no2_trop + no2_strat` against CAMS `tcno2` and
#   GEOS-CF `TotCol_NO2`. CAMS does not publish a tropospheric column.
#
# Caveats that matter here: neither model applies TEMPO's averaging kernels, so
# vertical sensitivity differs; TEMPO's stratospheric column is itself a model
# estimate; and both CAMS and GEOS-CF assimilate satellite NO₂, so they are not
# independent of the satellite record.

# %%
def tempo_on_grid(field: xr.DataArray, lat_c: np.ndarray, lon_c: np.ndarray,
                  min_coverage: float = MIN_COVERAGE) -> xr.DataArray:
    """Average TEMPO pixels into cells centred on a regular model grid."""
    lat_c, lon_c = np.asarray(lat_c, float), np.asarray(lon_c, float)
    dlat, dlon = lat_c[1] - lat_c[0], lon_c[1] - lon_c[0]
    lat2d, lon2d = np.meshgrid(field["lat"].values, field["lon"].values, indexing="ij")
    iy = np.floor((lat2d - (lat_c[0] - dlat / 2)) / dlat).astype(int)
    ix = np.floor((lon2d - (lon_c[0] - dlon / 2)) / dlon).astype(int)
    inside = (iy >= 0) & (iy < lat_c.size) & (ix >= 0) & (ix < lon_c.size)
    cell = np.where(inside, iy * lon_c.size + ix, -1).ravel()
    values = field.values.ravel()
    valid = np.isfinite(values) & (cell >= 0)
    size = lat_c.size * lon_c.size
    total = np.bincount(cell[cell >= 0], minlength=size).astype(float)
    count = np.bincount(cell[valid], minlength=size).astype(float)
    sums = np.bincount(cell[valid], weights=values[valid], minlength=size)
    # Cells cut by the TEMPO region edge are dropped: a partial cell is not
    # an average over the same area the model represents.
    expected = abs(dlat * dlon) / abs(
        float(np.diff(field["lat"].values).mean() * np.diff(field["lon"].values).mean())
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums / count
    keep = (total >= 0.9 * expected) & (count >= min_coverage * total)
    mean[~keep] = np.nan
    return xr.DataArray(
        mean.reshape(lat_c.size, lon_c.size),
        coords={"lat": lat_c, "lon": lon_c}, dims=("lat", "lon"), attrs=field.attrs,
    )


def model_at(ds: xr.Dataset, variable: str, when: pd.Timestamp, time_dim: str = "time") -> xr.DataArray:
    """Linear interpolation in time to a scan midpoint, never extrapolating."""
    times = pd.to_datetime(ds[time_dim].values)
    if not times.min() <= when <= times.max():
        raise KeyError(f"{variable} has no data around {when}")
    return ds[variable].interp({time_dim: np.datetime64(when)})


def stats(obs: np.ndarray, mod: np.ndarray) -> dict:
    mask = np.isfinite(obs) & np.isfinite(mod)
    o, m = obs[mask], mod[mask]
    if o.size < 3:
        return {"n": int(o.size)}
    return {
        "n": int(o.size),
        "obs_mean": float(o.mean()),
        "model_mean": float(m.mean()),
        "bias": float((m - o).mean()),
        "nmb_percent": float(100 * (m - o).sum() / o.sum()),
        "rmse": float(np.sqrt(((m - o) ** 2).mean())),
        "r": float(np.corrcoef(o, m)[0, 1]),
    }


cams = open_case("cams_fcst_no2.nc")
geoscf_columns = open_case("geoscf_ana_no2_columns.nc")

COLUMN_PAIRS = []  # (label, model dataset, model variable, TEMPO quantity)
if geoscf_columns is not None:
    COLUMN_PAIRS += [
        ("GEOS-CF analysis, tropospheric", geoscf_columns, "no2_trop", "trop"),
        ("GEOS-CF analysis, total", geoscf_columns, "no2_total", "total"),
    ]
if cams is not None:
    COLUMN_PAIRS.append(("CAMS forecast, total", cams, "no2_total", "total"))
print("Column comparisons:", [label for label, *_ in COLUMN_PAIRS] or "none staged")

# %%
def tempo_quantity(i: int, quantity: str) -> xr.DataArray:
    trop = screened(i, "no2_trop")
    if quantity == "trop":
        return trop
    return trop + screened(i, "no2_strat")


column_rows, column_fields = [], {}
for label, ds, variable, quantity in COLUMN_PAIRS:
    for i in range(len(scan_times)):
        when = scan_mid[i]
        try:
            mod = model_at(ds, variable, when)
        except KeyError as exc:
            print(f"  {label} scan {i}: {exc}")
            continue
        mod = mod.sel(lat=slice(region.lat_min, region.lat_max), lon=slice(region.lon_min, region.lon_max))
        obs = tempo_on_grid(tempo_quantity(i, quantity), mod.lat.values, mod.lon.values)
        column_fields[(label, i)] = (obs, mod)
        column_rows.append({"model": label, "scan": f"{when:%H:%M}Z", **stats(obs.values, mod.values)})

columns = pd.DataFrame(column_rows)
if not columns.empty:
    for col in ("obs_mean", "model_mean", "bias", "rmse"):
        columns[col] = columns[col] / 1e15
    print("Column statistics per scan (columns in 1e15 molecules cm-2):")
columns.round(2)

# %%
if not columns.empty:
    pooled = []
    for label, *_ in COLUMN_PAIRS:
        obs = np.concatenate([column_fields[k][0].values.ravel() for k in column_fields if k[0] == label])
        mod = np.concatenate([column_fields[k][1].values.ravel() for k in column_fields if k[0] == label])
        pooled.append({"model": label, **stats(obs / 1e15, mod / 1e15)})
    column_summary = pd.DataFrame(pooled).set_index("model")
else:
    column_summary = pd.DataFrame()
print("All scans pooled (1e15 molecules cm-2):")
column_summary.round(2)

# %% [markdown]
# ### 3.1 Maps for one scan
#
# Each row: TEMPO averaged to the model grid, the model, and model minus TEMPO.
# Blank cells were too cloudy or cut by the region edge.

# %%
if COLUMN_PAIRS:
    fig, axes = plt.subplots(len(COLUMN_PAIRS), 3, figsize=(15, 4.2 * len(COLUMN_PAIRS)),
                             squeeze=False, constrained_layout=True)
    for row, (label, *_rest) in enumerate(COLUMN_PAIRS):
        obs, mod = column_fields[(label, SCAN_INDEX)]
        vmax = np.nanpercentile(np.concatenate([obs.values.ravel(), mod.values.ravel()]) / 1e15, 98)
        diff = (mod - obs) / 1e15
        dmax = np.nanpercentile(np.abs(diff.values), 95)
        panels = [(obs / 1e15, "TEMPO on model grid", "magma_r", 0, vmax),
                  (mod / 1e15, label, "magma_r", 0, vmax),
                  (diff, "model − TEMPO", "RdBu_r", -dmax, dmax)]
        for ax, (field, title, cmap, lo, hi) in zip(axes[row], panels):
            mesh = ax.pcolormesh(field.lon, field.lat, field, cmap=cmap, vmin=lo, vmax=hi, shading="auto")
            ax.plot([west, east, east, west, west], [south, south, north, north, south], color="tab:blue", lw=1)
            ax.set_title(f"{title}  {scan_mid[SCAN_INDEX]:%H:%M}Z", fontsize=10)
            fig.colorbar(mesh, ax=ax, shrink=0.85, label="$10^{15}$ molecules cm$^{-2}$")

# %%
if COLUMN_PAIRS:
    fig, axes = plt.subplots(1, len(COLUMN_PAIRS), figsize=(5.2 * len(COLUMN_PAIRS), 4.8),
                             squeeze=False, constrained_layout=True)
    for ax, (label, *_rest) in zip(axes[0], COLUMN_PAIRS):
        keys = [k for k in column_fields if k[0] == label]
        for k in keys:
            obs, mod = column_fields[k]
            ax.scatter(obs.values.ravel() / 1e15, mod.values.ravel() / 1e15, s=8, alpha=0.5,
                       label=f"{scan_mid[k[1]]:%H:%M}Z")
        hi = np.nanmax([np.nanmax(column_fields[k][0].values) for k in keys]) / 1e15
        ax.plot([0, hi], [0, hi], "k--", lw=1)
        r = column_summary.loc[label, "r"] if "r" in column_summary else np.nan
        ax.set_title(f"{label}\nr = {r:.2f}", fontsize=10)
        ax.set_xlabel("TEMPO ($10^{15}$ molecules cm$^{-2}$)")
        ax.set_ylabel("model")
        ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=8, title="scan mid");

# %% [markdown]
# ### 3.2 New York City through the day
#
# The mean over model cells whose centre falls in the city box. At 0.25° that
# is a handful of cells; at CAMS's 0.4° it may be one or two. The city's plume
# is often narrower than a single cell, which alone biases coarse models low
# over the core and high downwind.

# %%
def box_mean(field: xr.DataArray) -> float:
    sel = field.sel(lat=slice(south, north), lon=slice(west, east))
    return float(sel.mean(skipna=True)) if sel.size else np.nan


city_rows = []
for (label, i), (obs, mod) in column_fields.items():
    mask = np.isfinite(obs)
    city_rows.append({"model": label, "time": scan_mid[i],
                      "TEMPO": box_mean(obs) / 1e15, "model_value": box_mean(mod.where(mask)) / 1e15,
                      "cells": int(np.isfinite(obs.sel(lat=slice(south, north), lon=slice(west, east))).sum())})
city_columns = pd.DataFrame(city_rows)

if not city_columns.empty:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, quantity in zip(axes, ("tropospheric", "total")):
        for label in city_columns.model.unique():
            if quantity not in label:
                continue
            rows = city_columns.query("model == @label").sort_values("time")
            (line,) = ax.plot(rows.time, rows.model_value, marker="s", label=label)
            # Each model gets its own TEMPO line: averaged onto that model's
            # cells (CAMS's 0.4-degree grid may put one cell over the city).
            ax.plot(rows.time, rows.TEMPO, marker="o", linestyle="--", color=line.get_color(),
                    alpha=0.8, label=f"TEMPO on {label.split(',')[0]} grid ({int(rows.cells.median())} cells)")
        ax.set_title(f"{CITY_NAME}: {quantity} NO$_2$ column")
        ax.set_ylabel("$10^{15}$ molecules cm$^{-2}$")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
city_columns.round({"TEMPO": 2, "model_value": 2})

# %% [markdown]
# ---
# ## 4. Chemical forecasts against surface monitors
#
# TEMPO sees the column; people breathe the surface. Here the models' lowest
# level is compared with EPA AirNow hourly NO₂ at the monitors inside the city
# box. On this day New York State did not report its New York City NO₂ monitors
# to AirNow, so the city is represented by the New Jersey sites across the
# Hudson — including Fort Lee, a near-road monitor that no 25 km model cell
# can be expected to match.
#
# Time matching: AirNow values are hour-beginning averages (hh:00–hh:59), which
# is exactly the averaging window of GEOS-CF's time-averaged hh:30 files. CAMS
# lowest-level NO₂ is instantaneous every 3 hours and is interpolated in time.
# OpenWeatherMap reports µg m⁻³, converted at 25 °C and 1 atm
# (1 ppb NO₂ = 1.88 µg m⁻³).

# %%
# AQS site ids are identifiers, not numbers; keep leading zeros and join cleanly.
airnow = pd.read_csv(CASE_DIR / "airnow_no2_hourly.csv", dtype={"station_id": str})
airnow["time_utc"] = pd.to_datetime(airnow["time_utc"], utc=True)
city_sites = airnow.loc[airnow.longitude.between(west, east) & airnow.latitude.between(south, north)]
city_sites = city_sites.rename(columns={"value": "AirNow"})
print(f"AirNow NO2 monitors in the {CITY_NAME} box: {sorted(city_sites.station_name.unique())}")

geoscf_surface = open_case("geoscf_fcst_surface_no2.nc")
owm_path = CASE_DIR / "owm_no2_history_nyc.csv"
owm = pd.read_csv(owm_path, dtype={"station_id": str}) if owm_path.is_file() else None
if owm is None:
    print("  not staged: owm_no2_history_nyc.csv")

UGM3_PER_PPB_NO2 = 46.0055 / 24.465  # 25 C, 1 atm


def point_series(ds, variable, frame, time_dim="time", centre=pd.Timedelta(0)):
    """Nearest-cell model value for each monitor row, interpolated in time."""
    out = np.full(len(frame), np.nan)
    times = pd.to_datetime(ds[time_dim].values)
    for n, row in enumerate(frame.itertuples(index=False)):
        when = row.time_utc.tz_convert(None) + centre
        if not times.min() <= when <= times.max():
            continue
        cell = ds[variable].sel(lat=row.latitude, lon=row.longitude, method="nearest")
        out[n] = float(cell.interp({time_dim: np.datetime64(when)}))
    return out


surface = city_sites.copy()
MODEL_COLUMNS = []
if geoscf_surface is not None:
    # AirNow hh:00 labels the hh:00-hh:59 mean; GEOS-CF stamps the same mean at hh:30.
    surface["GEOS-CF forecast"] = point_series(geoscf_surface, "no2_surface", surface,
                                               centre=pd.Timedelta(minutes=30))
    MODEL_COLUMNS.append("GEOS-CF forecast")
if cams is not None:
    surface["CAMS forecast"] = point_series(cams, "no2_surface", surface, time_dim="time_3h",
                                            centre=pd.Timedelta(minutes=30))
    MODEL_COLUMNS.append("CAMS forecast")
if owm is not None:
    owm_ppb = owm.assign(time_utc=pd.to_datetime(owm.time_utc, utc=True),
                         OpenWeatherMap=owm.no2_ugm3 / UGM3_PER_PPB_NO2)
    surface = surface.merge(owm_ppb[["station_id", "time_utc", "OpenWeatherMap"]],
                            on=["station_id", "time_utc"], how="left")
    MODEL_COLUMNS.append("OpenWeatherMap")

# Score every model on the same monitor-hours, or the rankings compare
# different parts of the day (OpenWeatherMap covers 24 h, GEOS-CF 13-20Z).
surface = surface.dropna(subset=MODEL_COLUMNS, how="any")
surface_stats = pd.DataFrame(
    [{"model": name, **stats(surface["AirNow"].to_numpy(float), surface[name].to_numpy(float))}
     for name in MODEL_COLUMNS]
).set_index("model") if MODEL_COLUMNS else pd.DataFrame()
print(f"Surface NO2 at {surface.station_id.nunique()} monitors, {surface.time_utc.min():%H}-"
      f"{surface.time_utc.max():%H}Z (ppb):")
surface_stats.round(2)

# %%
stations = sorted(surface.station_name.unique())
if stations:
    fig, axes = plt.subplots(1, len(stations), figsize=(4.6 * len(stations), 4), sharey=True,
                             squeeze=False, constrained_layout=True)
    for ax, station in zip(axes[0], stations):
        rows = surface.query("station_name == @station").sort_values("time_utc")
        ax.plot(rows.time_utc, rows.AirNow, color="k", marker="o", label="AirNow (observed)")
        for name in MODEL_COLUMNS:
            ax.plot(rows.time_utc, rows[name], marker=".", label=name)
        for mid in scan_mid:
            ax.axvline(mid.tz_localize("UTC"), color="0.8", linewidth=0.8, zorder=0)
        ax.set_title(station, fontsize=10)
        ax.tick_params(axis="x", rotation=40)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("surface NO$_2$ (ppb)")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle(f"{CITY_NAME} area surface NO$_2$ (grey lines: TEMPO scan midpoints)", fontsize=11)

# %% [markdown]
# ### 4.1 TEMPO at the monitors
#
# The same monitors against the TEMPO tropospheric column over them — the
# column/surface problem from notebook 05, on the day the models are scored.

# %%
tempo_trop = xr.concat([screened(i) for i in range(len(scan_times))], dim="time")
at_sites = city_sites.loc[city_sites.time_utc.dt.hour.isin(pd.to_datetime(scan_times).hour)].copy()
at_sites["tempo_no2_1e15"] = [
    float(tempo_trop.sel(time=np.datetime64(row.time_utc.tz_convert(None)), method="nearest")
          .sel(lat=row.latitude, lon=row.longitude, method="nearest")) / 1e15
    for row in at_sites.itertuples(index=False)
]
fig, ax = plt.subplots(figsize=(6, 4.5))
for station, rows in at_sites.groupby("station_name"):
    ax.scatter(rows.tempo_no2_1e15, rows.AirNow, label=station)
ax.set_xlabel("TEMPO tropospheric column ($10^{15}$ molecules cm$^{-2}$)")
ax.set_ylabel("AirNow surface NO$_2$ (ppb)")
ax.set_title("Column against surface at the city monitors")
ax.grid(alpha=0.3)
ax.legend(fontsize=8);

# %% [markdown]
# ---
# ## 5. One table
#
# Transport skill is dimensionless and positive is good. Column and surface
# rows are normalized mean bias and correlation against the observation.

# %%
scoreboard = []
for name in WIND_SOURCES:
    rows = transport.query("source == @name and level == @WIND_LEVEL")
    label = f"{name}{' (SYNTHETIC)' if name == MODEL_NAME and FCN_SYNTHETIC else ''}"
    scoreboard.append({"comparison": f"transport, {WIND_LEVEL} wind", "source": label,
                       "skill": rows.skill.mean(), "pairs": len(rows)})
for label, row in column_summary.iterrows():
    scoreboard.append({"comparison": "column vs TEMPO", "source": label,
                       "nmb_percent": row.get("nmb_percent"), "r": row.get("r"), "n": row.get("n")})
for label, row in surface_stats.iterrows():
    scoreboard.append({"comparison": "surface vs AirNow (city)", "source": label,
                       "nmb_percent": row.get("nmb_percent"), "r": row.get("r"), "n": row.get("n")})
scoreboard = pd.DataFrame(scoreboard).set_index(["comparison", "source"])
scoreboard.round(3)

# %% [markdown]
# ---
# ## 6. Save what you found

# %%
outputs = config.outputs / f"no2_intercomparison_{CASE_DATE}"
outputs.mkdir(parents=True, exist_ok=True)
transport.to_csv(outputs / "transport_skill.csv", index=False)
columns.to_csv(outputs / "column_stats_per_scan.csv", index=False)
city_columns.to_csv(outputs / "city_columns.csv", index=False)
surface.to_csv(outputs / "surface_city_monitors.csv", index=False)
scoreboard.to_csv(outputs / "scoreboard.csv")
(outputs / "run.json").write_text(json.dumps({
    "case_date": CASE_DATE, "region": region.name, "city_box": CITY_BOX,
    "tempo_source": tempo_uri, "fcn": fcn_summary, "fcn_synthetic": FCN_SYNTHETIC,
    "wind_level": WIND_LEVEL, "max_qa_flag": MAX_QA_FLAG,
    "max_cloud_fraction": MAX_CLOUD_FRACTION, "min_coverage": MIN_COVERAGE,
    "staged_files": sorted(manifest["files"]),
}, indent=2, default=str))
print(f"Wrote {outputs}")

# %% [markdown]
# ---
# ## 7. Optional: today's OpenWeatherMap forecast
#
# The OpenWeatherMap cells from the original version of this notebook, adapted
# for the Hub: the key comes from the `OPENWEATHERMAP_API_KEY` environment
# variable instead of Colab secrets. This is *today's* forecast, so it cannot
# be scored against 15 June; it is here to show the live service.

# %% tags=["optional"]
import urllib.parse
import urllib.request

OWM_KEY = os.environ.get("OPENWEATHERMAP_API_KEY")
LATITUDE, LONGITUDE = 40.7128, -74.0060  # New York City

if not OWM_KEY:
    print("Set OPENWEATHERMAP_API_KEY to run the live OpenWeatherMap forecast.")
else:
    query = urllib.parse.urlencode({"lat": LATITUDE, "lon": LONGITUDE, "appid": OWM_KEY})
    with urllib.request.urlopen(
        f"https://api.openweathermap.org/data/2.5/air_pollution/forecast?{query}", timeout=60
    ) as response:
        data = json.load(response)
    df_pollution = pd.DataFrame(
        [{"Datetime": pd.to_datetime(entry["dt"], unit="s", utc=True), "AQI": entry["main"]["aqi"],
          **entry["components"]} for entry in data["list"]]
    )
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].plot(df_pollution.Datetime, df_pollution.no2, marker="o", color="green")
    axes[0].set_title(f"Nitrogen dioxide forecast, lat {LATITUDE}, lon {LONGITUDE}")
    axes[0].set_ylabel("NO$_2$ (µg m$^{-3}$)")
    axes[1].plot(df_pollution.Datetime, df_pollution.o3, marker="x", color="orange")
    axes[1].set_title("Ozone forecast")
    axes[1].set_ylabel("O$_3$ (µg m$^{-3}$)")
    for ax in axes:
        ax.grid(alpha=0.3)
    display(df_pollution.head())

# %% [markdown]
# ---
# ## 8. Where to go next
#
# 1. **Apply the TEMPO averaging kernel** to the GEOS-CF profile (collection
#    `chm_inst_1hr_glo_L1440x721_v72` in the analysis) before comparing columns,
#    and see how much of the bias it removes.
# 2. **Interpolate winds in time** instead of taking the nearest lead, and check
#    whether FCN's 6-hour step is what separates it from GFS.
# 3. **Use a chemical model's own wind** (CAMS or GEOS-CF) for the advection:
#    does a model that is worse on NO₂ still move it well?
# 4. **Downscale the city.** Compare TEMPO at native 2 km against the model cell
#    it falls in and map the sub-grid variance the models cannot represent.
# 5. **Add DestinE** for a date inside its 15-day window (§2.6), where its 4.4 km
#    wind is the closest match to TEMPO's resolution of any source here.
# 6. **Hold out a day.** Stage a second date with
#    `scripts/stage_no2_intercomparison.py --date ...` and check whether the
#    rankings in §5 survive.
