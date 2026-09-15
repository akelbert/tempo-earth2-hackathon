# %% [markdown]
# # Bring your own site time series
#
# Change one filename and a few column names to turn a site CSV into a baseline
# experiment. The bundled file is synthetic; its purpose is to make the template
# executable before you substitute an approved dataset.

# %%
import matplotlib.pyplot as plt
import numpy as np

from tempo_earth2 import context

# Replace this with a path under the shared context-data root.
CSV_PATH = "demo/site_timeseries.csv"
TIME_COLUMN = "date"
TARGET = "carbon_flux_umol_m2_s"
FEATURES = ["water_level_m", "salinity_psu", "redox_mv", "temperature_c"]

data = context.read_csv(CSV_PATH, time_columns=(TIME_COLUMN,))
required = {TIME_COLUMN, TARGET, *FEATURES}
missing = required - set(data.columns)
if missing:
    raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

data = data.sort_values(TIME_COLUMN).reset_index(drop=True)
print(f"Rows: {len(data)}")
data.head()

# %% [markdown]
# ## Plot the target and inspect missing values

# %%
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data[TIME_COLUMN], data[TARGET], color="tab:green")
ax.set(title=TARGET, xlabel="time", ylabel=TARGET)
ax.grid(alpha=0.25)

data[[TARGET, *FEATURES]].isna().sum().rename("missing values")

# %% [markdown]
# ## Compare persistence with a linear model
#
# The first 70% of time trains the model; the final 30% tests it. This is more
# honest for prediction than randomly mixing later observations into training.

# %%
split = int(0.7 * len(data))
train = np.arange(len(data)) < split
test = ~train

data["linear_prediction"], _ = context.linear_baseline(
    data, FEATURES, TARGET, train=train
)
data["persistence_prediction"] = data[TARGET].shift(1)

def rmse(column):
    valid = test & data[column].notna() & data[TARGET].notna()
    return float(np.sqrt(np.mean((data.loc[valid, TARGET] - data.loc[valid, column]) ** 2)))

print(f"Persistence test RMSE: {rmse('persistence_prediction'):.3f}")
print(f"Linear-model test RMSE: {rmse('linear_prediction'):.3f}")

# %%
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(data.loc[test, TIME_COLUMN], data.loc[test, TARGET], label="observed", color="black")
ax.plot(
    data.loc[test, TIME_COLUMN],
    data.loc[test, "persistence_prediction"],
    label="persistence",
    linestyle="--",
)
ax.plot(
    data.loc[test, TIME_COLUMN],
    data.loc[test, "linear_prediction"],
    label="linear model",
)
ax.axvline(data.loc[split, TIME_COLUMN], color="grey", linewidth=1)
ax.set(title="Held-out final 30%", xlabel="time", ylabel=TARGET)
ax.legend()
ax.grid(alpha=0.25)

# %% [markdown]
# ## Make it yours
#
# 1. Upload or publish an approved CSV.
# 2. Point `CSV_PATH` at it.
# 3. Set the timestamp, target, and feature columns.
# 4. Keep the time split—or hold out a complete site or year.
# 5. Add TEMPO or Earth2 variables only after the baseline is working.
