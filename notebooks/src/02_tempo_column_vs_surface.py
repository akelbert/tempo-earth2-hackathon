# %% [markdown]
# # From a real TEMPO column to real surface NO₂
#
# TEMPO measures molecules through the atmospheric column. An EPA AQS monitor
# measures air near the ground. This notebook collocates six real TEMPO V04
# scans on 31 May 2026 with public hourly EPA observations, then fits two
# deliberately transparent baselines.
#
# Neither baseline is a validated exposure product. Their job is to make the
# unit, time, pixel/point, and vertical-representativeness problems concrete.
# Weather from FCN/HRRR is a natural next feature set, but we first establish
# what the observations alone do—and do not—support.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tempo_earth2 import catalog, context
from tempo_earth2.config import WorkshopConfig
from tempo_earth2.tempo import apply_quality_mask, open_tempo_source

config = WorkshopConfig.from_env()
matched_case = next(
    item for item in catalog.data_entries() if item["id"] == "tempo-aqs-2026-05-31"
)
case_uri = matched_case.get("workshop_uri")

try:
    tempo = open_tempo_source(case_uri) if case_uri else open_tempo_source(config.tempo_uri)
    print(f"TEMPO case:    {case_uri or config.tempo_uri}")
except Exception as exc:
    # Laptop/CI fixtures need not reproduce the event bucket hierarchy. They
    # still exercise the same array and collocation code using TEMPO_DATA_URI.
    print(f"Matched event case unavailable ({type(exc).__name__}); using TEMPO_DATA_URI")
    tempo = open_tempo_source(config.tempo_uri)
    print(f"TEMPO case:    {config.tempo_uri}")

case_date = str(tempo.time.values[0])[:10]
air = context.read_csv(f"aqs/{case_date}.csv")
air = air.query("parameter == 'NO2'").copy()

# Do not attach a midnight monitor value to the first afternoon satellite scan.
scan_start = pd.Timestamp(tempo.time.values.min()).tz_localize("UTC") - pd.Timedelta(
    minutes=31
)
scan_end = pd.Timestamp(tempo.time.values.max()).tz_localize("UTC") + pd.Timedelta(
    minutes=31
)
air = air.loc[air.time_utc.between(scan_start, scan_end)].copy()

print(f"Surface source: {air['source'].iloc[0]}")
print(f"UTC window:     {scan_start} to {scan_end}")
print(f"Stations:       {air.station_id.nunique()}")
print(f"Observations:   {len(air)}")

# %% [markdown]
# ## Collocate monitors with the nearest scan and pixel
#
# This is intentionally simple. A serious comparison should investigate
# collocation distance, representativeness, satellite averaging kernels, cloud
# screening, and the mismatch between an area-average column and a point
# monitor. The starter result makes those limitations visible rather than
# hiding them in a large model.

# %%
screened = tempo.assign(no2_trop=apply_quality_mask(tempo))
joined = context.nearest_grid_values(screened, air, ("no2_trop",))
joined["tempo_no2_1e15"] = joined["no2_trop"] / 1e15
joined["hour_utc"] = joined.time_utc.dt.hour + joined.time_utc.dt.minute / 60
joined[["station_name", "time_utc", "value", "tempo_no2_1e15"]].head()

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for station, group in joined.groupby("station_name"):
    axes[0].plot(group.time_utc, group.value, marker="o", label=station)
axes[0].set_ylabel("surface NO$_2$ (ppb)")
axes[0].set_title("EPA AQS hourly observations")
axes[0].tick_params(axis="x", rotation=30)

scatter = axes[1].scatter(
    joined.tempo_no2_1e15,
    joined.value,
    c=joined.hour_utc,
    cmap="viridis",
)
axes[1].set_xlabel("TEMPO column ($10^{15}$ molecules cm$^{-2}$)")
axes[1].set_ylabel("surface NO$_2$ (ppb)")
axes[1].set_title("Nearest-time, nearest-pixel comparison")
fig.colorbar(scatter, ax=axes[1], label="UTC hour")
fig.tight_layout()

# %% [markdown]
# ## Two transparent baselines
#
# Both are ordinary least squares. The second is allowed a smooth time-of-day
# cycle; it tests whether apparent column/surface agreement is actually just a
# shared diurnal pattern. These are in-sample diagnostics, so their scores are
# optimistic.

# %%
phase = 2 * np.pi * joined.hour_utc / 24
joined["hour_sin"] = np.sin(phase)
joined["hour_cos"] = np.cos(phase)

joined["column_only"], score_column = context.linear_baseline(
    joined, ["tempo_no2_1e15"], "value"
)
joined["column_plus_time"], score_time = context.linear_baseline(
    joined,
    ["tempo_no2_1e15", "hour_sin", "hour_cos"],
    "value",
)

print("Column only:       ", score_column)
print("Column + UTC time: ", score_time)

# %%
limit = [0, 1.05 * max(joined.value.max(), joined.column_plus_time.max())]
plt.figure(figsize=(6, 5))
plt.scatter(joined.value, joined.column_only, label="column only", alpha=0.7)
plt.scatter(joined.value, joined.column_plus_time, label="column + time", alpha=0.7)
plt.plot(limit, limit, "k--", linewidth=1, label="perfect")
plt.xlim(limit)
plt.ylim(limit)
plt.xlabel("observed surface NO$_2$ (ppb)")
plt.ylabel("baseline estimate (ppb)")
plt.title("Diagnostic baselines, not an exposure product")
plt.legend()
plt.grid(alpha=0.25)

# %% [markdown]
# ## Bring Earth-2 weather into the next experiment
#
# The next scientifically meaningful version should collocate boundary-layer
# and transport variables from a real source: FCN winds for a forecast
# experiment, or HRRR/reanalysis for a diagnostic comparison. Do not invent a
# boundary-layer height merely to make this introductory plot look complete.
#
# Try holding out one station, using a small satellite pixel average, adding
# FCN wind speed from notebook 01, or comparing FCN with HRRR. If performance
# collapses when a station is held out, the model learned station identity—not
# a general column-to-surface relationship.
