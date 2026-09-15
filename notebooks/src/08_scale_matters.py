# %% [markdown]
# # What disappears when real observing scales change?
#
# This notebook uses a real NASA HLS L30 surface-reflectance cutout over the
# Smithsonian Environmental Research Center (SERC) region on 4 June 2026. It
# calculates NDVI at the native 30 m grid, averages it to approximately TEMPO's
# kilometre scale, and measures the spatial information that averaging removes.

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle

from tempo_earth2 import context

scene = context.open_netcdf("hls/serc-2026-06-04.nc")
print(scene.attrs["source"])
print(scene.attrs["granule_id"])
print(f"Projected CRS: {scene.attrs['crs']}")
scene

# %% [markdown]
# ## Calculate clear-sky NDVI at the native HLS grid
#
# HLS Fmask bits 0–4 mark cirrus, cloud, adjacent cloud, cloud shadow, and
# snow/ice. We exclude those pixels and retain water so the land/water boundary
# remains visible. Surface-reflectance scaling was applied during staging.

# %%
clear = (scene.fmask.astype("uint8") & 0b00011111) == 0
valid_reflectance = (
    clear
    & (scene.red > 0)
    & (scene.nir > 0)
    & (scene.red <= 1.6)
    & (scene.nir <= 1.6)
)
ndvi = ((scene.nir - scene.red) / (scene.nir + scene.red)).where(valid_reflectance)
ndvi.name = "NDVI"
ndvi.attrs["long_name"] = "HLS normalized difference vegetation index"

print(f"Clear-pixel fraction: {float(clear.mean()):.1%}")
print(f"Analysis-pixel fraction: {float(valid_reflectance.mean()):.1%}")
plt.figure(figsize=(9, 5))
ndvi.plot(cmap="YlGn", vmin=0, vmax=0.9)
plt.title("Real HLS L30 NDVI over the SERC region (30 m)")

# %% [markdown]
# ## Average first, then return to the 30 m grid
#
# A TEMPO Level-3 pixel is roughly 2 km in this region, although its exact
# footprint differs from a square projected-grid average. Sixty HLS pixels make
# a simple 1.8 km teaching approximation. A coarse pixel is an average—not a
# blurry high-resolution measurement—and interpolation cannot restore the
# sub-pixel detail that was discarded.

# %%
factor = 60
coarse = ndvi.coarsen(y=factor, x=factor, boundary="trim").mean()
returned = coarse.interp(y=ndvi.y, x=ndvi.x, method="nearest")
residual = ndvi - returned

valid = np.isfinite(residual)
rmse = float(np.sqrt((residual.where(valid) ** 2).mean()))
print(f"HLS grid:         {ndvi.sizes['y']} x {ndvi.sizes['x']} at 30 m")
print(f"Approx. 1.8 km:   {coarse.sizes['y']} x {coarse.sizes['x']}")
print(f"Lost-detail RMSE: {rmse:.3f} NDVI")

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
ndvi.plot(ax=axes[0], cmap="YlGn", vmin=0, vmax=0.9)
axes[0].set_title("real HLS NDVI (30 m)")
returned.plot(ax=axes[1], cmap="YlGn", vmin=0, vmax=0.9)
axes[1].set_title("1.8 km mean, displayed at 30 m")
residual.plot(ax=axes[2], cmap="RdBu", vmin=-0.5, vmax=0.5)
axes[2].set_title("observed detail that was lost")

# %% [markdown]
# ## Put coarse Earth-2 weather beside fine HLS structure
#
# The FCN → Precipitation AFNO workflow supplies a real six-hour precipitation
# forecast from GFS initial conditions. Its 0.25° cells are far coarser than
# HLS. The local forecast is dry, so a continental panel shows that the model
# produced spatial structure elsewhere while a footprint panel makes the scale
# mismatch visible without inventing local variation.
# This cell uses Earth2Studio's native model, data, backend, and workflow APIs;
# notebook 03 explains the same interface as a compact model comparison.

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

initialization = datetime(2026, 6, 4, 12)
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
rain_field = rain.tp.isel(time=0, lead_time=-1) * 1000
west, south, east, north = map(float, scene.attrs["bbox_lonlat"].split(","))
regional = rain_field.sel(
    lon=slice(west - 1, east + 1),
    lat=slice(south - 1, north + 1),
)
continental = rain_field.sel(lon=slice(-130, -60), lat=slice(20, 55))
center_lon = (west + east) / 2
center_lat = (south + north) / 2
nearest_rain = rain_field.sel(lon=center_lon, lat=center_lat, method="nearest")
positive_rain = continental.values[continental.values > 0]
rain_vmax = max(float(np.nanquantile(positive_rain, 0.99)), 0.01)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
continental.plot(
    ax=axes[0],
    cmap="Blues",
    norm=PowerNorm(gamma=0.5, vmin=0, vmax=rain_vmax),
    cbar_kwargs={"label": "six-hour precipitation (mm)"},
)
axes[0].scatter(
    center_lon,
    center_lat,
    marker="*",
    s=140,
    color="tab:red",
    edgecolor="white",
    linewidth=0.5,
    label="HLS study area",
)
axes[0].set_title("Earth-2 precipitation context (local forecast is dry)")
axes[0].legend(loc="lower left")

cell_lon = float(nearest_rain.lon)
cell_lat = float(nearest_rain.lat)
axes[1].add_patch(
    Rectangle(
        (cell_lon - 0.125, cell_lat - 0.125),
        0.25,
        0.25,
        facecolor="aliceblue",
        edgecolor="tab:blue",
        linewidth=2,
        label="one 0.25° Earth-2 cell",
    )
)
axes[1].add_patch(
    Rectangle(
        (west, south),
        east - west,
        north - south,
        facecolor="none",
        edgecolor="tab:red",
        linewidth=2,
        label="HLS cutout",
    )
)
axes[1].scatter(cell_lon, cell_lat, marker="x", s=70, color="black")
axes[1].set_xlim(min(cell_lon - 0.16, west - 0.02), max(cell_lon + 0.16, east + 0.02))
axes[1].set_ylim(min(cell_lat - 0.16, south - 0.02), max(cell_lat + 0.16, north + 0.02))
axes[1].set_aspect("equal", adjustable="box")
axes[1].set_xlabel("longitude")
axes[1].set_ylabel("latitude")
axes[1].set_title("A weather cell versus the 30 m HLS cutout")
axes[1].legend(loc="upper right")

print("Earth2Studio workflow: FCN → PrecipitationAFNO")
print(f"GFS initialization:    {rain.attrs['initialization']} UTC")
print(
    f"Nearest Earth-2 cell center: {float(nearest_rain.lat):.2f}°, "
    f"{float(nearest_rain.lon):.2f}°"
)
print(f"Nearest-cell precipitation: {float(nearest_rain):.2f} mm over six hours")
print(f"Study-area maximum: {float(regional.max()):.3f} mm")
print("Interpretation: the model result is locally dry, not missing or failed.")
print("No interpolation here should be interpreted as added spatial information.")

# %% [markdown]
# ## Try one defensible extension
#
# Repeat the scale calculation for HLS shortwave-infrared indices, aggregate
# directly to actual TEMPO pixel footprints, or assemble multiple HLS dates
# around observed weather. A learned downscaler needs independent targets and
# spatially held-out validation; attractive high-resolution output alone is not
# evidence that the missing detail was recovered.
