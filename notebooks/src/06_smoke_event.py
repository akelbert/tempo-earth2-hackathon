# %% [markdown]
# # Follow a real wildfire-smoke event
#
# On 16 July 2026, smoke from fires in northwestern Ontario and northern
# Minnesota affected the Great Lakes and eastern United States. This notebook
# combines four real sources: NOAA HMS visible-smoke polygons and fire
# detections, preliminary AirNow PM₂.₅ observations, NASA TEMPO NO₂, and an
# Earth2Studio FCN wind forecast initialized from NOAA GFS.
#
# These layers make a transport hypothesis testable; they do not prove that a
# monitor response came from a particular fire. HMS maps visible smoke aloft,
# AirNow data are preliminary, and TEMPO NO₂ is not a direct smoke tracer.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon as PlotPolygon

from tempo_earth2 import catalog, context, forecast
from tempo_earth2.config import WorkshopConfig
from tempo_earth2.tempo import apply_quality_mask, open_tempo_source

CASE = "cases/2026-07-16-smoke"
EVENT_BOUNDS = (-100, 35, -65, 56)
config = WorkshopConfig.from_env()

air = context.read_csv(f"{CASE}/airnow_pm25_daily.csv", time_columns=())
air["date_local"] = pd.to_datetime(air.valid_date_local, format="%m/%d/%y")
fires = context.read_csv(f"{CASE}/hms_fires.csv")
smoke = context.read_json(f"{CASE}/hms_smoke.geojson")

case_entry = next(
    item for item in catalog.data_entries() if item["id"] == "smoke-2026-07-16"
)
try:
    tempo = open_tempo_source(case_entry["workshop_uri"])
    tempo_uri = case_entry["workshop_uri"]
except Exception as exc:
    # The compact CPU test fixture does not carry the 43 MB event Zarr. The
    # released/local workshop data do, and this fallback is printed visibly.
    print(f"Event TEMPO copy unavailable ({type(exc).__name__}); using TEMPO_DATA_URI")
    tempo = open_tempo_source(config.tempo_uri)
    tempo_uri = config.tempo_uri

print("Smoke/fire source: NOAA Hazard Mapping System")
print("PM2.5 source:      AirNow preliminary observations")
print(f"TEMPO source:      {tempo_uri}")
print(f"Smoke polygons:    {len(smoke['features']):,}")
print(f"Fire detections:   {len(fires):,}")
print(f"PM2.5 stations:    {air.station_id.nunique():,}")

# %% [markdown]
# ## Did surface particle pollution rise during the event?
#
# Each AirNow value below is a local-standard-day, 24-hour PM₂.₅ average. The
# median describes the broad region; the 90th percentile exposes the heavily
# affected tail without letting one site determine the story.

# %%
daily = air.groupby("date_local").value.agg(
    median="median",
    p90=lambda values: values.quantile(0.9),
    maximum="max",
    stations="count",
)
daily

# %%
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(daily.index, daily["median"], marker="o", label="regional median")
ax.plot(daily.index, daily.p90, marker="o", label="regional 90th percentile")
ax.plot(daily.index, daily.maximum, marker="o", alpha=0.55, label="site maximum")
ax.axvline(pd.Timestamp("2026-07-16"), color="black", linestyle="--", label="map day")
ax.set_ylabel(r"24-hour PM$_{2.5}$ ($\mu$g m$^{-3}$)")
ax.set_title("AirNow reports a strong regional particle-pollution episode")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## Map visible smoke, fires, monitors, and TEMPO context
#
# NOAA analysts assign light, medium, or heavy density to visible smoke
# polygons. Fire detections are sized by fire radiative power (FRP). TEMPO NO₂
# is shown only over its Northeast domain; combustion can emit NO₂, but this
# layer must not be interpreted as a quantitative smoke concentration.

# %%
density_color = {"Light": "#f6e58d", "Medium": "#f0932b", "Heavy": "#c0392b"}


def polygon_rings(geometry):
    coordinates = geometry["coordinates"]
    polygons = [coordinates] if geometry["type"] == "Polygon" else coordinates
    for polygon in polygons:
        yield np.asarray(polygon[0])


def add_smoke(ax, alpha=0.25):
    labels = set()
    for feature in smoke["features"]:
        density = feature["properties"]["density"]
        for ring in polygon_rings(feature["geometry"]):
            label = f"HMS {density.lower()} smoke" if density not in labels else None
            ax.add_patch(
                PlotPolygon(
                    ring,
                    closed=True,
                    facecolor=density_color.get(density, "grey"),
                    edgecolor="none",
                    alpha=alpha,
                    label=label,
                )
            )
            labels.add(density)


event_air = air.loc[air.date_local.eq(pd.Timestamp("2026-07-16"))].copy()
event_air["plot_value"] = event_air.value.clip(upper=300)
field = apply_quality_mask(tempo.isel(time=len(tempo.time) // 2)) / 1e15

fig, ax = plt.subplots(figsize=(12, 7))
add_smoke(ax)
mesh = ax.pcolormesh(
    field.lon,
    field.lat,
    field,
    shading="auto",
    cmap="Purples",
    alpha=0.62,
)
sampled_fires = fires.nlargest(5000, "frp_mw")
ax.scatter(
    sampled_fires.longitude,
    sampled_fires.latitude,
    s=3 + 2 * np.sqrt(sampled_fires.frp_mw.clip(lower=0)),
    marker="^",
    color="tab:red",
    alpha=0.35,
    linewidth=0,
    label="top 5,000 HMS fires by FRP",
)
points = ax.scatter(
    event_air.longitude,
    event_air.latitude,
    s=10 + event_air.plot_value,
    c=event_air.value,
    cmap="magma_r",
    vmin=0,
    vmax=200,
    edgecolor="black",
    linewidth=0.25,
    label="AirNow PM$_{2.5}$",
)
fig.colorbar(mesh, ax=ax, label="TEMPO NO$_2$ ($10^{15}$ molecules cm$^{-2}$)")
fig.colorbar(points, ax=ax, label=r"24-hour PM$_{2.5}$ ($\mu$g m$^{-3}$)")
ax.set(
    xlim=(EVENT_BOUNDS[0], EVENT_BOUNDS[2]),
    ylim=(EVENT_BOUNDS[1], EVENT_BOUNDS[3]),
    xlabel="longitude",
    ylabel="latitude",
)
ax.set_title("Observed layers on 16 July 2026")
ax.legend(loc="lower left", fontsize=8)

# %% [markdown]
# ## Does an Earth-2 wind forecast support the transport story?
#
# FCN starts from the 12 UTC GFS analysis. We select its 18 UTC 850 hPa wind,
# a useful free-tropospheric transport diagnostic, and overlay it on the HMS
# smoke. Agreement in direction supports plausibility; it still does not assign
# source contributions or represent winds throughout the full smoke layer.
# The convenience call below keeps this case study short; notebooks 02 and 03
# show the native Earth2Studio workflow explicitly.

# %% tags=["requires-gpu"]
earth2 = forecast.run_forecast(
    "2026-07-16T12:00",
    nsteps=2,
    model_name="FCN",
    variables=("u850", "v850"),
    verbose=False,
)
wind = forecast.select_valid_time(
    earth2.dataset, np.datetime64("2026-07-16T18:00")
)
wind = wind.sel(
    lon=slice(EVENT_BOUNDS[0], EVENT_BOUNDS[2]),
    lat=slice(EVENT_BOUNDS[1], EVENT_BOUNDS[3]),
)

fig, ax = plt.subplots(figsize=(12, 7))
add_smoke(ax, alpha=0.38)
stride = 8
ax.quiver(
    wind.lon.values[::stride],
    wind.lat.values[::stride],
    wind.u850.values[::stride, ::stride],
    wind.v850.values[::stride, ::stride],
    scale=380,
    width=0.0022,
    color="navy",
)
ax.scatter(
    sampled_fires.longitude,
    sampled_fires.latitude,
    s=5,
    color="tab:red",
    alpha=0.25,
    label="HMS fire detections",
)
ax.set(
    xlim=(EVENT_BOUNDS[0], EVENT_BOUNDS[2]),
    ylim=(EVENT_BOUNDS[1], EVENT_BOUNDS[3]),
    xlabel="longitude",
    ylabel="latitude",
)
ax.set_title(
    f"HMS visible smoke + Earth-2 FCN 850 hPa wind\nvalid {wind.attrs['valid_time']}"
)
ax.legend(loc="lower left")

print(f"Earth2Studio model: {earth2.model_name} on {earth2.device}")
print(f"GFS initialization: {earth2.init_time} UTC")
print(f"Forecast valid time: {wind.attrs['valid_time']}")

# %% [markdown]
# ## Try one defensible extension
#
# Compare the FCN wind with observed soundings or HRRR, quantify which monitors
# lie inside HMS polygons, or add aerosol optical depth. Preserve the temporal
# and vertical distinctions: daily surface PM₂.₅, instantaneous visible smoke,
# a forecast pressure-level wind, and a TEMPO trace-gas column are not
# interchangeable measurements.
