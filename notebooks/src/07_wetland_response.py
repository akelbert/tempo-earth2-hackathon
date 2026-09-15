# %% [markdown]
# # Real coastal water levels: what is not explained by the tide?
#
# This notebook uses NOAA CO-OPS observations and astronomical tide predictions
# from Annapolis, Maryland, around the real 31 May 2026 TEMPO case. Subtracting
# the prediction exposes a simple non-tidal residual that can flag periods worth
# investigating for winds, pressure, runoff, or modeled rainfall.
#
# Annapolis is regional Chesapeake Bay context—not a GCReW marsh sensor and not
# evidence of wetland carbon response. Replace or join it with approved GCReW
# observations before making site-level ecological claims.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle

from tempo_earth2 import context

observed = context.read_csv(
    "coops/annapolis-water-level-2026-05-25-2026-06-02.csv"
)
predicted = context.read_csv(
    "coops/annapolis-tide-predictions-2026-05-25-2026-06-02.csv"
)

print(observed.source.iloc[0])
print(f"Station: {observed.station_name.iloc[0]} ({observed.station.iloc[0]})")
print(f"Datum: {observed.datum.iloc[0]}; units: metres")
print(f"Observed rows: {len(observed):,}; prediction rows: {len(predicted):,}")

# %% [markdown]
# ## Align observation and prediction
#
# Both series are six-minute data relative to mean sea level. An exact timestamp
# join is appropriate here; for two independent instruments we would instead
# declare and inspect an explicit matching tolerance.

# %%
water = observed[["time_utc", "value"]].rename(columns={"value": "observed_m"})
tide = predicted[["time_utc", "value"]].rename(columns={"value": "predicted_m"})
joined = water.merge(tide, on="time_utc", validate="one_to_one")
joined["residual_m"] = joined.observed_m - joined.predicted_m

if len(joined) != len(observed):
    raise ValueError("Observation/prediction timestamps did not align completely")
joined.describe()

# %%
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
axes[0].plot(joined.time_utc, joined.observed_m, label="observed", color="black")
axes[0].plot(
    joined.time_utc,
    joined.predicted_m,
    label="astronomical prediction",
    color="tab:blue",
    alpha=0.8,
)
axes[0].set_ylabel("water level (m MSL)")
axes[0].legend()
axes[0].grid(alpha=0.25)

axes[1].plot(joined.time_utc, joined.residual_m, color="tab:orange")
axes[1].axhline(0, color="grey", linewidth=1)
axes[1].set_ylabel("observed − predicted (m)")
axes[1].set_xlabel("UTC")
axes[1].grid(alpha=0.25)
fig.suptitle("NOAA Annapolis water level and non-tidal residual")
fig.tight_layout()

# %% [markdown]
# ## Find periods worth explaining
#
# A daily maximum avoids treating every six-minute sample as an independent
# event. The residual alone does not identify its cause.

# %%
daily = (
    joined.set_index("time_utc")
    .resample("1D")
    .agg(
        observed_max_m=("observed_m", "max"),
        residual_max_m=("residual_m", "max"),
        residual_mean_m=("residual_m", "mean"),
    )
    .reset_index()
)
daily["date"] = daily.time_utc.dt.date
daily.sort_values("residual_max_m", ascending=False)

# %%
case_day = pd.Timestamp("2026-05-31", tz="UTC")
case = joined.loc[
    joined.time_utc.between(case_day, case_day + pd.Timedelta(days=1), inclusive="left")
]
print(f"31 May maximum observed level: {case.observed_m.max():.3f} m MSL")
print(f"31 May maximum non-tidal residual: {case.residual_m.max():+.3f} m")
print(f"31 May RMS residual: {float(np.sqrt(np.mean(case.residual_m**2))):.3f} m")

# %% [markdown]
# ## Add live Earth-2 precipitation
#
# Run the coupled FCN → Precipitation AFNO workflow from a real GFS
# initialization. We sample its six-hour accumulation near Annapolis and place
# it beside the observed non-tidal residual over the same window. This happens
# to be a dry Chesapeake forecast. The continental view verifies that the model
# produced a structured precipitation field rather than a failed/blank output,
# while preserving the scientifically important local zero.
# The cell uses Earth2Studio's native model, data, backend, and workflow APIs.
# Notebook 03 explains the same interface as a compact model comparison.

# %% tags=["requires-gpu"]
from datetime import datetime

import torch
from earth2studio import run
from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.dx import PrecipitationAFNO
from earth2studio.models.px import FCN

from tempo_earth2.forecast import to_xarray

if not torch.cuda.is_available():
    raise RuntimeError("This live Earth-2 workflow requires the workshop GPU")

initialization = datetime(2026, 5, 31, 12)
prognostic = FCN.load_model(FCN.load_default_package())
diagnostic = PrecipitationAFNO.load_model(
    PrecipitationAFNO.load_default_package()
)
rain_backend = run.diagnostic(
    [initialization],
    1,
    prognostic,
    diagnostic,
    GFS(),
    ZarrBackend(file_name=None),
    device=torch.device("cuda"),
    verbose=False,
)
rain = to_xarray(rain_backend, ("tp",))
rain.attrs["initialization"] = initialization.isoformat()
rain_field = rain.tp.isel(time=0, lead_time=-1)
station_lon = -76.4817
station_lat = 38.9833
station_rain_mm = float(
    rain_field.sel(lon=station_lon, lat=station_lat, method="nearest") * 1000
)
valid_time = pd.Timestamp(
    rain.valid_time.isel(time=0, lead_time=-1).values
).tz_localize("UTC")
rain_window = joined.loc[
    joined.time_utc.between(valid_time - pd.Timedelta(hours=6), valid_time)
]

print("Earth2Studio workflow: FCN → PrecipitationAFNO")
print(f"GFS initialization:    {rain.attrs['initialization']} UTC")
print(f"Forecast valid time:   {valid_time}")
print(f"Annapolis rainfall:    {station_rain_mm:.2f} mm over six hours")
print(f"Mean residual in window: {rain_window.residual_m.mean():+.3f} m")

chesapeake_rain = (
    rain_field.sel(lon=slice(-78, -75), lat=slice(37.5, 40.5)) * 1000
)
continental_rain = (
    rain_field.sel(lon=slice(-130, -60), lat=slice(20, 55)) * 1000
)
positive_rain = continental_rain.values[continental_rain.values > 0]
rain_vmax = max(float(np.nanquantile(positive_rain, 0.99)), 0.01)

print(f"Chesapeake-area maximum: {float(chesapeake_rain.max()):.3f} mm")
print("Interpretation: this six-hour forecast is locally dry; rainfall is not a")
print("plausible explanation for the simultaneous water-level residual by itself.")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
axes[0].plot(rain_window.time_utc, rain_window.residual_m, color="tab:orange")
axes[0].axhline(0, color="grey", linewidth=1)
axes[0].set_title("Observed Annapolis residual during forecast window")
axes[0].set_ylabel("observed − predicted water level (m)")
axes[0].set_xlabel("UTC")
axes[0].grid(alpha=0.25)
axes[0].text(
    0.03,
    0.95,
    f"Earth-2 rain near station: {station_rain_mm:.2f} mm",
    transform=axes[0].transAxes,
    va="top",
    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
)

continental_rain.plot(
    ax=axes[1],
    cmap="Blues",
    norm=PowerNorm(gamma=0.5, vmin=0, vmax=rain_vmax),
    cbar_kwargs={"label": "six-hour precipitation (mm)"},
)
axes[1].add_patch(
    Rectangle(
        (-78, 37.5),
        3,
        3,
        fill=False,
        edgecolor="tab:red",
        linewidth=2,
        label="Chesapeake window",
    )
)
axes[1].scatter(
    station_lon,
    station_lat,
    marker="*",
    s=120,
    color="tab:red",
    edgecolor="white",
    linewidth=0.5,
    label="Annapolis",
)
axes[1].set_title("Structured precipitation elsewhere; Chesapeake is dry")
axes[1].legend(loc="lower left")

# %% [markdown]
# For GCReW, bring in approved site water level, salinity, redox, and
# carbon/nitrogen measurements, extend the modeled/observed weather record, and
# retain a leave-one-site or leave-one-year validation design. The regional
# Annapolis comparison is hypothesis generation, not site-level attribution.
