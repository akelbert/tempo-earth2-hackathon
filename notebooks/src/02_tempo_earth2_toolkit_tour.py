# %% [markdown]
# # Toolkit tour: `tempo_earth2` and Earth2Studio
#
# The workshop helper package handles paths, TEMPO conventions, grid alignment,
# and small scientific diagnostics. Earth2Studio itself remains the model API:
# this notebook loads and runs FCN using NVIDIA's public classes rather than a
# workshop-specific model wrapper. Every result remains an ordinary pandas or
# xarray object.
#
# Use this notebook as a short reference. Notebook 01 catalogs the available
# data, the science notebooks show complete questions, and notebook 03 compares
# the installed models in more detail.

# %%
from datetime import UTC
import inspect

import numpy as np
import pandas as pd

import tempo_earth2.advect as advect
import tempo_earth2.catalog as catalog
import tempo_earth2.context as context
import tempo_earth2.forecast as forecast
import tempo_earth2.plots as plots
import tempo_earth2.tempo as tempo
from tempo_earth2.config import WorkshopConfig, describe_environment

config = WorkshopConfig.from_env()
print(f"TEMPO source:  {config.tempo_uri}")
print(f"Context root:  {config.context_uri}")
print(f"Output folder: {config.outputs}")

# %% [markdown]
# ## What lives in each module?
#
# The table is built from the installed functions, so its names and one-line
# descriptions stay in sync with the environment rather than being copied into
# prose. The full signatures remain available in `helper_reference`.

# %%
groups = {
    "catalog": [catalog.as_frame, catalog.data_entries, catalog.model_entries],
    "tempo": [tempo.load_tempo_scan, tempo.subset, tempo.apply_quality_mask],
    "context": [context.read_csv, context.open_netcdf, context.nearest_grid_values],
    "forecast": [
        forecast.nearest_init_time,
        forecast.steps_to_cover,
        forecast.select_valid_time,
        forecast.regrid_to,
    ],
    "advect": [advect.wind_speed, advect.experiment],
    "plots": [plots.plot_no2, plots.plot_no2_with_wind],
}

rows = []
for module, functions in groups.items():
    for function in functions:
        summary = (inspect.getdoc(function) or "").split("\n", maxsplit=1)[0]
        rows.append(
            {
                "module": module,
                "function": function.__name__,
                "purpose": summary,
                "signature": f"{function.__name__}{inspect.signature(function)}",
            }
        )

helper_reference = pd.DataFrame(rows)
helper_reference[["module", "function", "purpose"]]

# %% [markdown]
# To inspect any helper in more detail, use normal Python discovery:

# %%
print(inspect.signature(tempo.load_tempo_scan))
print(inspect.getdoc(tempo.load_tempo_scan).split("Parameters", maxsplit=1)[0])

# %% [markdown]
# ## Configuration: write one notebook that works locally and in JupyterHub
#
# `WorkshopConfig.from_env()` resolves local paths or `gs://` URIs without
# hard-coding a project, bucket, or username. Save derived results beneath
# `config.outputs`; that directory is inside the user's persistent home in the
# deployed Hub.

# %%
environment = describe_environment()
pd.Series(
    {
        "Python": environment.get("python"),
        "Earth2Studio": environment.get("earth2studio"),
        "CUDA visible": environment.get("cuda_available", False),
        "GPU": environment.get("gpu_name", "none"),
        "TEMPO": config.tempo_uri,
        "context": config.context_uri,
        "outputs": str(config.outputs),
    },
    name="resolved value",
)

# %% [markdown]
# ## TEMPO: normalized coordinates and one quality-screening call
#
# `load_tempo_scan` accepts a local path or Cloud Storage URI. It normalizes the
# coordinate names to `time`, `lat`, and `lon`, uses longitudes from −180° to
# 180°, and can select a named workshop region. The returned object is ordinary
# xarray, so all normal xarray selection, grouping, plotting, and export methods
# remain available.

# %%
scan = tempo.load_tempo_scan(config.tempo_uri, region="northeast")
no2 = tempo.apply_quality_mask(scan)

print(f"Dataset sizes:       {dict(scan.sizes)}")
print(f"Variables:           {list(scan.data_vars)}")
print(f"NO2 units:           {no2.attrs.get('units', 'see source metadata')}")
print(f"Finite after screen: {float(np.isfinite(no2).mean()):.1%}")

# %% [markdown]
# Context datasets follow the same pattern. These functions return familiar
# pandas or xarray objects and automatically resolve the configured context
# root:
#
# ```python
# air_quality = context.read_csv("aqs/2026-05-31.csv")
# hls = context.open_netcdf("hls/serc-2026-06-04.nc")
# print(context.data_uri("cases/2026-07-16-smoke/manifest.json"))
# ```
#
# Use `catalog.data_entries(["smoke"])` to discover sources by theme and
# `catalog.as_frame("data")` for the full access table.

# %% [markdown]
# ## Earth-2: plan the initialization before running a model
#
# Earth2Studio weather models initialize at fixed UTC boundaries. Always round
# **down**, so the initialization cannot see observations from the future, and
# calculate enough model steps to reach the observation time.

# %%
observed_time = pd.Timestamp(scan.time.values).to_pydatetime().replace(tzinfo=UTC)
init_time = forecast.nearest_init_time(observed_time)
nsteps = forecast.steps_to_cover(init_time, observed_time)

print(f"TEMPO observation: {observed_time:%Y-%m-%d %H:%M} UTC")
print(f"Model initialized: {init_time:%Y-%m-%d %H:%M} UTC")
print(f"Steps to cover it: {nsteps} (six hours per FCN step)")

# %% [markdown]
# ## Minimal live FCN interface
#
# This is the native Earth2Studio pattern: load a model package, choose a data
# source and output backend, then call a workflow runner. Requesting only the
# variables needed for the question is the most effective way to limit memory
# and output size. `ZarrBackend(file_name=None)` keeps this short result in
# memory; use a path beneath `config.outputs` for a longer run that should
# survive a kernel restart.
#
# This cell requires the workshop GPU. There is intentionally no CPU fallback.

# %% tags=["requires-gpu"]
import torch
from earth2studio import run
from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import FCN

if not torch.cuda.is_available():
    raise RuntimeError("This live Earth-2 example requires an NVIDIA GPU")

device = torch.device("cuda")
variables = np.array(["u10m", "v10m", "t2m"])
model = FCN.load_model(FCN.load_default_package())
output = run.deterministic(
    [init_time.replace(tzinfo=None)],
    nsteps,
    model,
    GFS(),
    ZarrBackend(file_name=None),
    output_coords={"variable": variables},
    device=device,
    verbose=False,
)

fcn = forecast.to_xarray(output, variables)
weather = forecast.select_valid_time(fcn, observed_time)
weather_on_tempo = forecast.regrid_to(weather, no2.lat, no2.lon)
speed = advect.wind_speed(weather_on_tempo.u10m, weather_on_tempo.v10m)

print(f"Model class:         {type(model).__name__}")
print(f"Earth2Studio device: {device}")
print(f"Dataset dimensions:  {dict(fcn.sizes)}")
print(f"Selected valid time: {weather.attrs['valid_time']}")
print(f"Mean 10 m wind:      {float(speed.mean()):.1f} m/s")
plots.plot_no2_with_wind(no2, weather_on_tempo.u10m, weather_on_tempo.v10m)

# %% [markdown]
# The important return values are:
#
# - `output`: Earth2Studio's IO backend containing the requested model fields;
# - `fcn`: the same fields converted to an xarray `Dataset` with `time`,
#   `lead_time`, `valid_time`, `lat`, and `lon` coordinates;
# - `weather`: the forecast slice nearest the requested observation time; and
# - `weather_on_tempo`: interpolated onto TEMPO's grid for arithmetic or
#   overlays. Interpolation aligns pixels—it does not create finer weather
#   information.

# %% [markdown]
# ## Additional installed model interfaces
#
# DLWP and PrecipitationAFNO use the same native Earth2Studio components as FCN.
# All three checkpoints are installed and L4-validated. The executable
# comparisons are in notebook 03; the essential calls are:
#
# ```python
# from earth2studio.models.px import DLWP
#
# dlwp = DLWP.load_model(DLWP.load_default_package())
# dlwp_output = run.deterministic(
#     [init_time.replace(tzinfo=None)], 1, dlwp, GFS(),
#     ZarrBackend(file_name=None),
#     output_coords={"variable": np.array(["t2m"])},
#     device=torch.device("cuda"),
# )
#
# # FCN atmospheric forecast followed by the precipitation diagnostic
# from earth2studio.models.dx import PrecipitationAFNO
#
# diagnostic = PrecipitationAFNO.load_model(
#     PrecipitationAFNO.load_default_package()
# )
# rain_output = run.diagnostic(
#     [init_time.replace(tzinfo=None)], 1, model, diagnostic, GFS(),
#     ZarrBackend(file_name=None), device=torch.device("cuda"),
# )
# rain = forecast.to_xarray(rain_output, ("tp",))
# rain_mm = rain.tp * 1000  # Earth2Studio returns `tp` in metres
# ```
#
# Both calls require a workshop GPU. See notebook 03 for the executable
# comparison and the model catalog for the exact-image validation record.

# %%
print("Installed prognostics:", ", ".join(forecast.SUPPORTED_PROGNOSTICS))
print("Installed diagnostics:", ", ".join(forecast.SUPPORTED_DIAGNOSTICS))

catalog.as_frame("models")[[
    "id", "kind", "tier", "live_inference", "science_uses"
]]

# %% [markdown]
# ## The boundary to remember
#
# - **Earth2Studio:** `models`, `data`, `io`, and `run` perform model inference.
# - **`tempo_earth2`:** configuration, catalogs, TEMPO/context loading,
#   quality control, time/grid alignment, plots, and workshop diagnostics.
#
# This keeps model code transferable to other Earth2Studio projects. Whichever
# interface you extend, preserve initialization, lead time, variables, units,
# checkpoint version, and input-data provenance in every saved result.
