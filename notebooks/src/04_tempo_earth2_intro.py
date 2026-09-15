# %% [markdown]
# # Where does the NO₂ go? TEMPO observations meet an Earth-2 forecast
#
# **Run notebooks 00–03 first.** This notebook begins the science path and
# assumes the environment check passed.
#
# ---
#
# ## The question
#
# NASA's TEMPO instrument measures the nitrogen dioxide column over North
# America roughly once an hour during daylight. NO₂ near the surface comes
# mostly from combustion — traffic, power generation, industry — so a TEMPO
# image of the northeast corridor is close to a photograph of where the engines
# are.
#
# But the image changes between scans, and not because the engines moved. Over
# an hour or two, near-surface NO₂ is mostly **transported**: the wind carries
# it. Chemistry and deposition matter, but on this timescale the wind dominates.
#
# That gives us a testable claim:
#
# > If we take the NO₂ field TEMPO measured at one scan and push every pixel
# > downwind using a **forecast** wind field, we should get closer to the next
# > TEMPO scan than if we had assumed nothing moved.
#
# The forecast wind comes from **Earth2Studio**, NVIDIA's AI weather modeling
# framework. We initialize a global model from operational GFS analysis, run it
# forward, and read out the near-surface wind over the TEMPO domain.
#
# Nothing in this notebook is a research result. It is a scaffold: it gets a
# real satellite retrieval and a real AI forecast into the same array, on the
# same grid, at the same time, and produces a number you can argue with. What
# you do next is the hackathon.
#
# ## What you will do
#
# 1. Load a staged TEMPO NO₂ scan and look at it.
# 2. Run an Earth-2 forecast from GFS initial conditions.
# 3. Put the forecast wind onto the TEMPO grid and overlay the two.
# 4. Check that NO₂ and wind speed are related the way ventilation says they
#    should be.
# 5. Advect the NO₂ field with the forecast wind and score it against the next
#    scan.
# 6. Pick an idea from the list at the bottom and break something.
#
# ## Running time
#
# On a dedicated GPU: about three to five minutes, most of it the first model
# load. If no GPU is visible, report the platform incident and do not run the
# inference section until service is restored.

# %% [markdown]
# ---
# ## 0. Settings
#
# Everything you are likely to change is in this one cell. Re-run the notebook
# from here after editing it.

# %%
# --- What to look at ----------------------------------------------------------
REGION = "northeast"      # northeast | chicago | losangeles | texas | conus
SCAN_INDEX = 1            # which staged scan to use as t0 (0 = first)
SCAN_GAP = 1              # how many scans ahead t1 is (1 = the next hour)

# --- Data quality -------------------------------------------------------------
MAX_QA_FLAG = 0           # 0 keeps only normal-quality retrievals
MAX_CLOUD_FRACTION = 0.2  # raise towards 1.0 to keep more (cloudier) pixels

# --- The forecast -------------------------------------------------------------
MODEL_NAME = "FCN"        # see tempo_earth2.forecast.SUPPORTED_PROGNOSTICS
WIND_LEVEL = "10m"        # "10m" | "100m" | "850"

# --- Optional teaching/reference data ----------------------------------------
# Instructors may set this to True for a demonstration or a fast analysis loop.
# It is not the event's availability fallback; attendee servers still require a GPU.
USE_REFERENCE_FORECAST = False

# %%
import warnings

import numpy as np
import xarray as xr

from tempo_earth2 import advect, forecast, plots
from tempo_earth2.config import WorkshopConfig
from tempo_earth2.tempo import REGIONS, apply_quality_mask, open_tempo_source, subset

warnings.filterwarnings("ignore", category=RuntimeWarning)

config = WorkshopConfig.from_env()
config.ensure_dirs()
region = REGIONS[REGION]

print(f"Region:      {region.name}  {region.bbox}")
print(f"TEMPO data:  {config.tempo_uri}")
print(f"Model:       {MODEL_NAME}")
print(f"GPU:         {'yes' if forecast.cuda_available() else 'NO - report incident'}")

if not forecast.cuda_available() and not USE_REFERENCE_FORECAST:
    raise RuntimeError(
        "No CUDA device is visible. Report this platform incident and work on "
        "non-GPU project tasks until GPU service is restored."
    )

# %% [markdown]
# ---
# ## 1. The observation
#
# The staged TEMPO copy is a regional subset of the
# [TEMPO gridded NO₂ product](https://asdc.larc.nasa.gov/project/TEMPO), already
# cut down from the ~850 MB full-disk granules. It carries the tropospheric and
# stratospheric NO₂ columns, a quality flag, and an effective cloud fraction.
#
# We want the **tropospheric** column: the stratospheric part is large, smooth,
# and has nothing to do with what is happening at street level.

# %%
tempo = subset(open_tempo_source(config.tempo_uri), region=region)
tempo

# %%
scan_times = tempo["time"].values
print(f"{len(scan_times)} scans staged:\n")
for i, t in enumerate(scan_times):
    marker = ""
    if i == SCAN_INDEX:
        marker = "  <- t0"
    elif i == SCAN_INDEX + SCAN_GAP:
        marker = "  <- t1"
    print(f"  [{i}] {np.datetime_as_string(t, unit='m')}Z{marker}")

if SCAN_INDEX + SCAN_GAP >= len(scan_times):
    raise IndexError(
        f"SCAN_INDEX ({SCAN_INDEX}) + SCAN_GAP ({SCAN_GAP}) is past the last "
        f"staged scan ({len(scan_times) - 1}). Lower one of them."
    )

t0 = scan_times[SCAN_INDEX]
t1 = scan_times[SCAN_INDEX + SCAN_GAP]
dt_hours = float((t1 - t0) / np.timedelta64(1, "h"))
print(f"\nGap between t0 and t1: {dt_hours:.2f} hours")

# %% [markdown]
# ### Quality screening
#
# Two screens are applied, and both matter more than they look:
#
# * **`main_data_quality_flag == 0`** keeps only normal retrievals.
# * **Effective cloud fraction ≤ 0.2** removes pixels where the retrieval is
#   really seeing cloud top rather than the boundary layer.
#
# Screened pixels become `NaN`, not zero. They are *missing*, not *clean*, and
# treating them as clean air is one of the easiest ways to get a wrong answer
# out of this dataset. Notice, in the map below, how much of the domain that
# removes — and that the removed area moves between scans.

# %%
no2_t0 = apply_quality_mask(
    tempo.isel(time=SCAN_INDEX),
    max_flag=MAX_QA_FLAG,
    max_cloud_fraction=MAX_CLOUD_FRACTION,
)
no2_t1 = apply_quality_mask(
    tempo.isel(time=SCAN_INDEX + SCAN_GAP),
    max_flag=MAX_QA_FLAG,
    max_cloud_fraction=MAX_CLOUD_FRACTION,
)

valid_fraction = float(np.isfinite(no2_t0.values).mean())
print(f"Valid pixels at t0: {valid_fraction:.1%} of the domain")
print(f"Grid:               {dict(no2_t0.sizes)}")

if valid_fraction < 0.15:
    print(
        "\n  Fewer than 15% of pixels survived screening. Try a different scan,\n"
        "  or raise MAX_CLOUD_FRACTION in the settings cell."
    )

# %%
plots.plot_no2(
    no2_t0,
    title=f"TEMPO tropospheric NO$_2$  {np.datetime_as_string(t0, unit='m')}Z",
)

# %% [markdown]
# ---
# ## 2. The forecast
#
# Earth2Studio models are initialized from an analysis and stepped forward on a
# fixed timestep. `FCN` (FourCastNet) uses 6 hours, so the valid initialization
# times are 00, 06, 12 and 18 UTC.
#
# We snap **down** from the TEMPO scan time to the most recent initialization.
# Rounding to the *nearest* would sometimes pick an initialization later than
# the observation, which quietly turns a forecast comparison into a reanalysis
# comparison — the model would have already seen the answer.

# %%
from datetime import datetime, timezone

t0_dt = datetime.fromisoformat(np.datetime_as_string(t0, unit="s")).replace(
    tzinfo=timezone.utc
)
t1_dt = datetime.fromisoformat(np.datetime_as_string(t1, unit="s")).replace(
    tzinfo=timezone.utc
)

init_time = forecast.nearest_init_time(t0_dt)
nsteps = forecast.steps_to_cover(init_time, t1_dt)

print(f"TEMPO t0:          {t0_dt:%Y-%m-%d %H:%M}Z")
print(f"Model init:        {init_time:%Y-%m-%d %H:%M}Z")
print(f"Lead time to t0:   {(t0_dt - init_time).total_seconds() / 3600:.1f} hours")
print(f"Forecast steps:    {nsteps}  ({nsteps * 6} hours)")

# %% [markdown]
# Now run it. The event image already contains the reviewed checkpoint. Model
# initialization is still the slow part, but it must not require a download.
#
# We ask for only the eight wind and near-surface variables we need. The model
# computes all 26 internally either way, but writing only what we use keeps the
# output store about ten times smaller.
#
# This science notebook uses `tempo_earth2.forecast.run_forecast` to keep the
# transport narrative readable. Notebook 02 shows the equivalent native
# Earth2Studio model/data/IO/run calls; the wrapper does not define a different
# model API.

# %%
store = config.outputs / f"{MODEL_NAME.lower()}_{init_time:%Y%m%dT%H%M}Z.zarr"

if USE_REFERENCE_FORECAST:
    path = forecast.precomputed_forecast_path(config, MODEL_NAME, init_time)
    print(f"Reading precomputed forecast: {path}")
    fx = forecast.load_precomputed(path)
    run_summary = {"source": "precomputed", "store": str(path)}
else:
    result = forecast.run_forecast(
        init_time.replace(tzinfo=None),
        nsteps=nsteps,
        model_name=MODEL_NAME,
        variables=forecast.TRANSPORT_VARIABLES,
        store_path=store,
    )
    fx = result.dataset
    run_summary = result.summary()
    print()
    for key, value in run_summary.items():
        print(f"  {key:<14} {value}")

fx

# %% [markdown]
# ### Picking the right lead time
#
# The forecast has several lead times; we want the one whose **valid time** is
# closest to the TEMPO scan. For the advection experiment we want the wind
# during the interval, so we ask for the midpoint between t0 and t1.

# %%
midpoint = t0_dt + (t1_dt - t0_dt) / 2

fx_t0 = forecast.select_valid_time(fx, t0_dt)
fx_mid = forecast.select_valid_time(fx, midpoint)

print(f"For t0        ({t0_dt:%H:%M}Z): lead {fx_t0.attrs['lead_hours']:.0f} h, "
      f"valid {fx_t0.attrs['valid_time'][:16]}Z")
print(f"For midpoint  ({midpoint:%H:%M}Z): lead {fx_mid.attrs['lead_hours']:.0f} h, "
      f"valid {fx_mid.attrs['valid_time'][:16]}Z")
print(
    "\nNote the mismatch: the model steps every 6 hours, TEMPO scans every hour."
    "\nThat temporal gap is a real limitation of this comparison, and one of the"
    "\nmore interesting things you could attack."
)

# %% [markdown]
# ---
# ## 3. Onto the same grid
#
# TEMPO L3 is about 0.02° — roughly 2 km. `FCN` is 0.25° — roughly 25 km. To
# overlay them we interpolate the model wind onto the TEMPO grid.
#
# Be clear about what that does and does not do: it puts the wind on the same
# pixels as the NO₂, and it adds **no information whatsoever**. A 25 km wind
# field cannot resolve the sea breeze along the Connecticut shore or the flow
# around Manhattan. Where the advection experiment below fails, this is one of
# the first places to look.

# %%
u_name, v_name = {
    "10m": ("u10m", "v10m"),
    "100m": ("u100m", "v100m"),
    "850": ("u850", "v850"),
}[WIND_LEVEL]

available = set(fx.data_vars)
if u_name not in available:
    raise KeyError(
        f"{MODEL_NAME} did not produce {u_name}. Available: {sorted(available)}. "
        "Change WIND_LEVEL in the settings cell."
    )

wind = forecast.regrid_to(
    fx_mid[[u_name, v_name]], no2_t0["lat"].values, no2_t0["lon"].values
)
u, v = wind[u_name], wind[v_name]
speed = advect.wind_speed(u, v)

print(f"Wind level:       {WIND_LEVEL} ({u_name}, {v_name})")
print(f"Model grid:       {fx.sizes['lat']} x {fx.sizes['lon']}")
print(f"TEMPO grid:       {no2_t0.sizes['lat']} x {no2_t0.sizes['lon']}")
print(f"Mean wind speed:  {float(speed.mean()):.1f} m/s")
print(f"Max wind speed:   {float(speed.max()):.1f} m/s")

# %%
plots.plot_no2_with_wind(
    no2_t0,
    u,
    v,
    title=(
        f"TEMPO NO$_2$ {np.datetime_as_string(t0, unit='m')}Z "
        f"with {MODEL_NAME} {WIND_LEVEL} wind "
        f"(+{fx_mid.attrs['lead_hours']:.0f} h)"
    ),
)

# %% [markdown]
# **Look at this figure before going further.** Do the plumes trail downwind of
# the cities you can identify? If they trail *upwind*, something is wrong with a
# sign, a longitude convention, or the lead time — and the score in §5 will be
# meaningless until you find it.

# %% [markdown]
# ---
# ## 4. A sanity check: ventilation
#
# Before testing anything clever, test something that has to be true.
#
# Emissions are roughly fixed on the hour timescale. If the wind is stronger,
# the same emissions get spread over more air, so the column above any given
# point should be **lower**. Plotting median NO₂ against forecast wind speed
# should give a decreasing curve.
#
# If it does not, the model wind and the TEMPO retrieval are not describing the
# same atmosphere, and there is no point continuing to §5.

# %%
relationship = advect.ventilation_relationship(no2_t0, speed, bins=12)
plots.plot_ventilation(relationship)

# %%
mask = np.isfinite(no2_t0.values) & np.isfinite(speed.values)
correlation = np.corrcoef(no2_t0.values[mask], speed.values[mask])[0, 1]
print(f"Pearson correlation, NO2 column vs wind speed: {correlation:+.3f}")
print(
    "\nExpect a negative value. A weak one is normal - wind speed is far from"
    "\nthe only thing setting the column, and the emission field is extremely"
    "\nuneven. A positive value is worth investigating."
)

# %% [markdown]
# ---
# ## 5. The experiment: advect and score
#
# Take the NO₂ field at t0, move every pixel downwind by the forecast wind for
# `dt_hours`, and compare against what TEMPO actually saw at t1.
#
# The comparison is against **persistence** — the assumption that nothing moved.
# Persistence is a genuinely strong baseline over one hour, which is exactly why
# it is the right thing to beat.
#
# ```
# skill = 1 - RMSE(advected) / RMSE(persistence)
# ```
#
# Positive means the forecast wind carried real information about where the NO₂
# went. Zero means it was no better than doing nothing. Negative means it made
# things worse.

# %%
advected, skill = advect.experiment(
    no2_t0,
    no2_t1,
    u,
    v,
    dt_hours=dt_hours,
    wind_level=f"{MODEL_NAME} {WIND_LEVEL}",
    substeps=4,
)

print(skill)

# %%
plots.plot_advection_triptych(no2_t0, advected, no2_t1, skill=skill)

# %% [markdown]
# ### Reading the number honestly
#
# A small positive skill score is the expected outcome, and it is not very
# impressive. Before you decide the model is good or bad, work out how much of
# the score is physics and how much is bookkeeping:
#
# * **The wind is 6-hourly, the observation hourly.** We used a wind valid up to
#   three hours from the interval we applied it to.
# * **The wind is 25 km, the observation 2 km.** All of the fine structure that
#   makes a plume look like a plume is unresolved.
# * **One level stands in for a whole column.** The 10 m wind is not the wind
#   that moved the NO₂ at 800 m.
# * **Missing data moves.** Cloud-screened pixels differ between t0 and t1, so
#   the scored sample is not a fixed set of locations.
# * **No chemistry, no emission, no deposition, no vertical mixing.** NO₂ has a
#   lifetime of hours; over one hour that is a modest error, over three it is
#   not.
#
# Each of those is a hackathon project.

# %% [markdown]
# ### How sensitive is it?
#
# One number is an anecdote. Run the same experiment at each available wind
# level and see whether the ranking is stable.

# %%
import pandas as pd

comparison = []
for level, (un, vn) in {
    "10m": ("u10m", "v10m"),
    "100m": ("u100m", "v100m"),
    "850": ("u850", "v850"),
}.items():
    if un not in fx.data_vars:
        continue
    w = forecast.regrid_to(
        fx_mid[[un, vn]], no2_t0["lat"].values, no2_t0["lon"].values
    )
    try:
        _, s = advect.experiment(
            no2_t0, no2_t1, w[un], w[vn], dt_hours=dt_hours, wind_level=level
        )
        comparison.append(
            {
                "level": level,
                "skill": round(s.skill_score, 4),
                "rmse_advected": s.rmse_advected,
                "rmse_persistence": s.rmse_persistence,
                "n_valid": s.n_valid,
            }
        )
    except ValueError as exc:
        print(f"  {level}: {exc}")

pd.DataFrame(comparison).set_index("level")

# %% [markdown]
# ---
# ## 6. Interactive exploration (optional)
#
# The cell below gives you a slider over the staged scans. It is a convenience,
# not part of the analysis — if widgets are misbehaving in your browser, skip
# it, nothing downstream depends on it.

# %% tags=["optional"]
import ipywidgets as widgets
from IPython.display import display

import matplotlib.pyplot as plt

_scan_slider = widgets.IntSlider(
    value=SCAN_INDEX, min=0, max=len(scan_times) - 1, description="scan", continuous_update=False
)
_cloud_slider = widgets.FloatSlider(
    value=MAX_CLOUD_FRACTION, min=0.05, max=1.0, step=0.05,
    description="max cloud", continuous_update=False,
)
_output = widgets.Output()


def _explore(_=None):
    with _output:
        _output.clear_output(wait=True)
        print("_explore() called")  # diagnostic: confirm the callback fires at all
        i = _scan_slider.value
        field = apply_quality_mask(
            tempo.isel(time=i),
            max_flag=MAX_QA_FLAG,
            max_cloud_fraction=_cloud_slider.value,
        )
        stamp = np.datetime_as_string(scan_times[i], unit="m")
        valid = float(np.isfinite(field.values).mean())
        plots.plot_no2(field, title=f"{stamp}Z   ({valid:.0%} valid pixels)")
        plt.show()


_scan_slider.observe(_explore, names="value")
_cloud_slider.observe(_explore, names="value")
display(widgets.HBox([_scan_slider, _cloud_slider]), _output)
_explore()

# %% [markdown]
# ### Linked views with Glue
#
# Glue lets you select a region in one view and see the same pixels highlighted
# in another — for example, select the high-NO₂ pixels in the scatter and see
# where they sit on the map.
#
# **Make your selection in the scatter panel, not the map panel.** The map
# viewer's own selection tool currently crashes in-browser (a confirmed bug in
# the bqplot-image-gl/bqplot-gl WebGL interaction code, not in this notebook —
# there is no newer library release to fix it against yet). The scatter panel
# uses plain bqplot, not that WebGL layer, so its selection tool works, and
# because both panels share the same underlying Glue `Data` object, a
# selection made there still highlights the matching pixels on the map.

# %% tags=["optional"]
try:
    from glue.core import Data
    from glue_jupyter import jglue

    # Coarsen before handing it to Glue. An unresponsive browser tab is worse
    # than no linked view at all, and the full TEMPO grid will produce one.
    step = max(1, no2_t0["lon"].size // 300)
    coarse = dict(lat=slice(None, None, step), lon=slice(None, None, step))

    # Both fields go into a *single* glue Data object. That is what makes the
    # views linked: selecting pixels on the map selects the same rows in the
    # scatter, because they are the same rows.
    linked = Data(
        label="TEMPO NO2 and Earth-2 wind",
        no2=np.asarray(no2_t0.isel(**coarse).values, dtype=float),
        wind_speed=np.asarray(speed.isel(**coarse).values, dtype=float),
    )

    app = jglue()
    app.data_collection.append(linked)
    app.imshow(data=linked)
    app.scatter2d(x="no2", y="wind_speed", data=linked)

    print(
        "Two linked viewers created. Use the selection tool in the SCATTER "
        "panel\n(not the map) to highlight the matching pixels on the map - "
        "the map's own\nselection tool has a known WebGL crash, tracked in "
        "the design document."
    )
except Exception as exc:
    print(f"Glue view unavailable ({type(exc).__name__}: {exc}).")
    print("This is optional - nothing above depends on it.")

# %% [markdown]
# ---
# ## 7. Save what you found
#
# Your home directory persists across server restarts. Everything else does not.

# %%
import json

outputs = config.outputs
outputs.mkdir(parents=True, exist_ok=True)

record = {
    "region": region.name,
    "t0": np.datetime_as_string(t0, unit="s"),
    "t1": np.datetime_as_string(t1, unit="s"),
    "dt_hours": dt_hours,
    "model": MODEL_NAME,
    "wind_level": WIND_LEVEL,
    "max_qa_flag": MAX_QA_FLAG,
    "max_cloud_fraction": MAX_CLOUD_FRACTION,
    "valid_fraction_t0": valid_fraction,
    "no2_wind_correlation": float(correlation),
    "skill": {
        "score": skill.skill_score,
        "rmse_advected": skill.rmse_advected,
        "rmse_persistence": skill.rmse_persistence,
        "n_valid": skill.n_valid,
    },
    "level_comparison": comparison,
    "forecast_run": run_summary,
}

record_path = outputs / "intro_result.json"
record_path.write_text(json.dumps(record, indent=2, default=str))
print(f"Wrote {record_path}")


def _bare(da):
    """Drop scalar time/lead_time coordinates before merging.

    `no2_t0` carries the t0 scan time, `no2_t1` carries t1, and the wind carries
    its own valid time. Merging them into one Dataset without dropping those
    would be a coordinate conflict, so keep only lat and lon.
    """
    return da.reset_coords(drop=True)


result_ds = xr.Dataset(
    {
        "no2_t0": _bare(no2_t0),
        "no2_t1": _bare(no2_t1),
        "no2_advected": _bare(advected),
        u_name: _bare(u),
        v_name: _bare(v),
    },
    attrs={
        "t0": np.datetime_as_string(t0, unit="s"),
        "t1": np.datetime_as_string(t1, unit="s"),
        "model": MODEL_NAME,
        "wind_level": WIND_LEVEL,
        "skill_score": skill.skill_score,
    },
)
result_path = outputs / "intro_fields.zarr"
result_ds.to_zarr(result_path, mode="w", consolidated=True, zarr_format=2)
print(f"Wrote {result_path}")

# %% [markdown]
# ---
# ## 8. Where to go next
#
# Pick one. Each is a day's work, and each has a version that fits in an
# afternoon.
#
# **Make the comparison fairer**
#
# 1. Use a model with a shorter timestep, or interpolate the wind in time
#    between lead times rather than snapping to the nearest.
# 2. Advect with a pressure-weighted average of several wind levels instead of
#    one, and see whether the best-performing weighting tells you anything about
#    where the NO₂ actually sits.
# 3. Score only pixels that are valid in *both* scans, and separately report how
#    much of the domain that discards. Does the skill score change?
#
# **Add missing physics**
#
# 4. Add a first-order chemical loss term with a photolysis-dependent lifetime
#    and refit. Does an NO₂ lifetime fall out of the residual?
# 5. Add divergence: the current scheme conserves the tracer, but a converging
#    wind field should pile it up.
# 6. Identify point sources (power plants) and estimate emission rates from the
#    downwind plume, in the style of the standard TEMPO/TROPOMI flux method.
#
# **Push on the forecast**
#
# 7. Swap `MODEL_NAME` for another Earth2Studio model and compare skill. Do
#    models that score better on standard weather metrics also move NO₂ better?
# 8. Run an ensemble (`earth2studio.run.ensemble`) and turn the wind spread into
#    an uncertainty band on the advected NO₂.
# 9. Use `CorrDiff` or `StormCastCONUS` to downscale the wind to something
#    closer to the TEMPO resolution before advecting.
#
# **Go the other way**
#
# 10. Use TEMPO as validation for the *model* rather than the reverse: where the
#     advection fails systematically, is that a known model bias in the
#     boundary-layer wind?
#
# ---
#
# ### Getting unstuck
#
# * `restore-workshop-notebooks 04_tempo_earth2_intro.ipynb` — clean copy back,
#   your version moved aside, not deleted.
# * `reset-user-environment` — undo a `pip install` that broke imports.
# * `00_environment_check.ipynb` — run it again and paste the output into your
#   support request.
