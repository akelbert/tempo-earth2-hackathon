# %% [markdown]
# # A wetland response baseline
#
# Can hydrology help predict redox or carbon exchange at a site not used for
# training? This notebook uses a clearly synthetic GCReW-shaped table to teach
# time-series alignment and leave-one-site-out validation.

# %%
import matplotlib.pyplot as plt
import numpy as np

from tempo_earth2 import context

wetland = context.read_csv("demo/wetland_timeseries.csv")
print(wetland.source.iloc[0])
print(f"Sites: {wetland.site_id.nunique()}, rows: {len(wetland)}")
wetland.head()

# %% [markdown]
# ## Look before fitting

# %%
site = wetland.query("site_id == 'MARSH_A'")
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
axes[0].plot(site.date, site.water_level_m, color="tab:blue")
axes[0].set_ylabel("water level (m)")
axes[1].plot(site.date, site.redox_mv, color="tab:orange")
axes[1].set_ylabel("redox (mV)")
axes[2].plot(site.date, site.carbon_flux_umol_m2_s, color="tab:green")
axes[2].set_ylabel("carbon flux")
axes[2].set_xlabel("date")
fig.suptitle("Hydrology and biogeochemistry move together—but not identically")
fig.tight_layout()

# %% [markdown]
# ## Hold out an entire site
#
# Randomly splitting rows would leak each site's characteristic behavior into
# both train and test sets. Instead, the model never sees `MARSH_D` while fitting.

# %%
features = ["water_level_m", "salinity_psu", "redox_mv", "temperature_c"]
target = "carbon_flux_umol_m2_s"
held_out = "MARSH_D"
train = wetland.site_id != held_out

wetland["prediction"], training_metrics = context.linear_baseline(
    wetland, features, target, train=train
)
test = wetland.site_id == held_out
test_rmse = np.sqrt(
    np.mean((wetland.loc[test, target] - wetland.loc[test, "prediction"]) ** 2)
)

print("Fit diagnostics across all rows:", training_metrics)
print(f"Held-out {held_out} RMSE: {test_rmse:.3f}")

# %%
held = wetland.loc[test]
plt.figure(figsize=(10, 4.5))
plt.plot(held.date, held[target], label="observed", color="black")
plt.plot(held.date, held.prediction, label="predicted", color="tab:red")
plt.ylabel("carbon flux ($\mu$mol m$^{-2}$ s$^{-1}$)")
plt.title(f"Leave-one-site-out check: {held_out}")
plt.legend()
plt.grid(alpha=0.25)

# %% [markdown]
# ## Try one change
#
# Predict redox instead of carbon flux, hold out a year rather than a site, add
# lagged water level, or replace this fixture with an approved GCReW extract.
# Always keep the held-out group biologically meaningful.
