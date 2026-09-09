# %% [markdown]
# # Follow a smoke event
#
# This teaching fixture contains a moving PM2.5 pulse and fire detections. It is
# synthetic, so the expected ordering is known. Replace it with a synchronized
# TEMPO–FIRMS–AQS case before drawing scientific conclusions.

# %%
import matplotlib.pyplot as plt
import numpy as np

from tempo_earth2 import context
from tempo_earth2.config import WorkshopConfig
from tempo_earth2.tempo import apply_quality_mask, open_tempo_source

config = WorkshopConfig.from_env()
air = context.read_csv("demo/smoke_air_quality.csv")
fires = context.read_csv("demo/fire_detections.csv")
tempo = open_tempo_source(config.tempo_uri)

print(air.source.iloc[0])
print(fires.source.iloc[0])

# %% [markdown]
# ## When did the smoke peak?

# %%
fig, ax = plt.subplots(figsize=(10, 4.5))
for station, group in air.groupby("station_name"):
    ax.plot(group.time_utc, group.pm25_ug_m3, marker="o", label=station)
ax.set_ylabel("PM$_{2.5}$ ($\mu$g m$^{-3}$)")
ax.set_title("The synthetic plume arrives later downwind")
ax.tick_params(axis="x", rotation=30)
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()

peaks = air.loc[air.groupby("station_name").pm25_ug_m3.idxmax()]
peaks[["station_name", "time_utc", "pm25_ug_m3"]].sort_values("time_utc")

# %% [markdown]
# ## Put the instruments on one map
#
# The TEMPO layer supplies atmospheric context. The fire and monitor points do
# not prove causation; transport time and wind direction must agree too.

# %%
field = apply_quality_mask(tempo.isel(time=len(tempo.time) // 2)) / 1e15

fig, ax = plt.subplots(figsize=(9, 6))
mesh = ax.pcolormesh(field.lon, field.lat, field, shading="auto", cmap="magma_r")
fire_sizes = 15.0 + 1.2 * fires.frp_mw
ax.scatter(
    fires.longitude,
    fires.latitude,
    s=fire_sizes,
    marker="^",
    color="tab:red",
    edgecolor="white",
    label="fire detections",
)
stations = air.drop_duplicates("station_id")
ax.scatter(
    stations.longitude,
    stations.latitude,
    s=70,
    color="tab:cyan",
    edgecolor="black",
    label="PM2.5 monitors",
)
for row in stations.itertuples():
    ax.text(row.longitude + 0.12, row.latitude, row.station_name, fontsize=8)
fig.colorbar(mesh, ax=ax, label="TEMPO NO$_2$ ($10^{15}$ molecules cm$^{-2}$)")
ax.set(xlabel="longitude", ylabel="latitude", title="One event, several observing systems")
ax.legend()

# %% [markdown]
# ## A first transport calculation
#
# Estimate the speed between the central and downwind monitor from their peak
# times. This is only a scale check, but impossible speeds immediately reveal a
# bad matching assumption.

# %%
central = peaks.set_index("station_name").loc["Central"]
downwind = peaks.set_index("station_name").loc["Downwind"]
hours = (downwind.time_utc - central.time_utc).total_seconds() / 3600

latitude_km = (downwind.latitude - central.latitude) * 111.0
longitude_km = (
    (downwind.longitude - central.longitude)
    * 111.0
    * np.cos(np.radians((downwind.latitude + central.latitude) / 2))
)
distance_km = np.hypot(latitude_km, longitude_km)
speed_ms = distance_km * 1000 / (hours * 3600)

print(f"peak separation: {hours:.1f} hours")
print(f"station distance: {distance_km:.0f} km")
print(f"implied speed:    {speed_ms:.1f} m/s")

# %% [markdown]
# ## Try one change
#
# Compare the implied speed with Earth2Studio or HRRR wind, use a different
# TEMPO trace gas, or calculate whether the PM2.5 pulse broadens downwind.
