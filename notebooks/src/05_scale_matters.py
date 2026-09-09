# %% [markdown]
# # What disappears when scales change?
#
# TEMPO pixels and high-resolution imagery answer different questions. This
# notebook calculates a vegetation index, averages it to a coarse grid, and
# measures the information that averaging removes. The reflectance scene is a
# synthetic HLS-shaped teaching fixture.

# %%
import matplotlib.pyplot as plt
import numpy as np

from tempo_earth2 import context

scene = context.open_netcdf("demo/land_surface.nc")
print(scene.attrs.get("source"))
scene

# %% [markdown]
# ## Calculate NDVI at the native grid

# %%
ndvi = (scene.nir - scene.red) / (scene.nir + scene.red)
ndvi.name = "NDVI"
ndvi.attrs["long_name"] = "normalized difference vegetation index"

plt.figure(figsize=(9, 5))
ndvi.plot(cmap="YlGn", vmin=0, vmax=0.8)
plt.title("Fine-grid vegetation structure")

# %% [markdown]
# ## Average first, then return to the fine grid
#
# A coarse pixel is an average, not a blurry high-resolution measurement. Once
# sub-pixel variation has been averaged away, interpolation cannot recreate it.

# %%
factor = 15
coarse = ndvi.coarsen(lat=factor, lon=factor, boundary="trim").mean()
returned = coarse.interp(lat=ndvi.lat, lon=ndvi.lon, method="nearest")
residual = ndvi - returned

valid = np.isfinite(residual)
rmse = float(np.sqrt((residual.where(valid) ** 2).mean()))
print(f"Fine grid:   {ndvi.sizes['lat']} x {ndvi.sizes['lon']}")
print(f"Coarse grid: {coarse.sizes['lat']} x {coarse.sizes['lon']}")
print(f"Lost-detail RMSE: {rmse:.3f} NDVI")

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
ndvi.plot(ax=axes[0], cmap="YlGn", vmin=0, vmax=0.8)
axes[0].set_title("fine NDVI")
returned.plot(ax=axes[1], cmap="YlGn", vmin=0, vmax=0.8)
axes[1].set_title("coarse mean, displayed finely")
residual.plot(ax=axes[2], cmap="RdBu", vmin=-0.35, vmax=0.35)
axes[2].set_title("detail that was lost")

# %% [markdown]
# ## Try one change
#
# Change `factor`, compare red-edge NDVI, or aggregate a real HLS cutout to the
# actual TEMPO grid. A machine-learning downscaler can learn relationships with
# high-resolution predictors, but it cannot discover unobserved detail without
# assumptions and independent validation.
