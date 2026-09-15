# %% [markdown]
# # Run more than FCN: an Earth-2 model tasting menu
#
# This notebook performs live inference rather than displaying fabricated model
# output. FCN and DLWP start from the same real NOAA GFS analysis, then the
# supported v1 Precipitation AFNO diagnostic turns FCN fields into six-hour
# accumulated precipitation.
#
# **Release status:** all three model checkpoints are baked into the workshop
# image and passed the exact-image NVIDIA L4 release test without runtime model
# downloads. Availability is not a claim that one model is scientifically
# superior to another.

# %% tags=["requires-gpu"]
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle

from earth2studio import run
from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.dx import PrecipitationAFNO
from earth2studio.models.px import DLWP, FCN

from tempo_earth2.forecast import to_xarray

if not torch.cuda.is_available():
    raise RuntimeError(
        "No NVIDIA GPU is visible. Run this from the workshop GPU server; "
        "there is intentionally no CPU fallback."
    )

INIT_TIME = datetime(2026, 5, 31, 12)
REGION = {"lat": slice(36, 45), "lon": slice(-80, -66)}
print(torch.cuda.get_device_name(0))
print(f"Real GFS initialization: {INIT_TIME:%Y-%m-%d %H:%M} UTC")


def run_temperature(model_class):
    """Run one native Earth2Studio prognostic and retain only 2 m temperature."""
    model = model_class.load_model(model_class.load_default_package())
    output = run.deterministic(
        [INIT_TIME],
        1,
        model,
        GFS(),
        ZarrBackend(file_name=None),
        output_coords={"variable": np.array(["t2m"])},
        device=torch.device("cuda"),
        verbose=False,
    )
    return model, to_xarray(output, ("t2m",))

# %% [markdown]
# ## Ask two AI weather models the same question
#
# Both models receive GFS initial conditions through Earth2Studio. We request
# only 2 m temperature because it is a common output; this keeps the comparison
# fair and the in-memory result small. Agreement is not truth, and disagreement
# is not automatically error—it tells us where verification is worth doing.

# %% tags=["requires-gpu"]
temperature_models = {}
temperature_runs = {}
for model_class in (FCN, DLWP):
    name = model_class.__name__
    model, dataset = run_temperature(model_class)
    temperature_models[name] = model
    temperature_runs[name] = dataset
    print(name, dict(dataset.sizes), list(dataset.data_vars))

# %% tags=["requires-gpu"]
lead = np.timedelta64(6, "h")
fcn_t2m = temperature_runs["FCN"].t2m.sel(lead_time=lead).squeeze().sel(**REGION)
dlwp_t2m = temperature_runs["DLWP"].t2m.sel(lead_time=lead).squeeze().sel(**REGION)
difference = dlwp_t2m - fcn_t2m

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
fcn_t2m.plot(ax=axes[0], cmap="coolwarm", cbar_kwargs={"label": "K"})
axes[0].set_title("FCN +6 h temperature")
dlwp_t2m.plot(ax=axes[1], cmap="coolwarm", cbar_kwargs={"label": "K"})
axes[1].set_title("DLWP +6 h temperature")
difference.plot(ax=axes[2], cmap="RdBu_r", center=0, cbar_kwargs={"label": "K"})
axes[2].set_title("DLWP minus FCN")

print(f"Regional mean difference: {float(difference.mean()):+.2f} K")
print(f"Regional RMS difference:  {float(np.sqrt((difference**2).mean())):.2f} K")

# %% [markdown]
# ## Couple FCN to a precipitation diagnostic
#
# Precipitation AFNO consumes twenty FCN atmospheric fields and produces
# six-hour accumulated total precipitation. These are model results initialized
# from a real analysis—not observations. A serious project should verify them
# against gauges, radar, or a gridded precipitation product.

# %% tags=["requires-gpu"]
diagnostic = PrecipitationAFNO.load_model(
    PrecipitationAFNO.load_default_package()
)
precipitation_output = run.diagnostic(
    [INIT_TIME],
    1,
    temperature_models["FCN"],
    diagnostic,
    GFS(),
    ZarrBackend(file_name=None),
    device=torch.device("cuda"),
    verbose=False,
)
precipitation = to_xarray(precipitation_output, ("tp",))
precip_mm = (
    precipitation.tp.sel(lead_time=lead).squeeze()
    .sel(lon=slice(-130, -60), lat=slice(20, 55))
    * 1000
)
precip_mm.attrs.update(units="mm", long_name="six-hour accumulated precipitation")
positive = precip_mm.values[precip_mm.values > 0]
vmax = max(float(np.nanquantile(positive, 0.99)), 0.01)

plt.figure(figsize=(9, 5))
axis = plt.gca()
precip_mm.plot(
    ax=axis,
    cmap="Blues",
    norm=PowerNorm(gamma=0.5, vmin=0, vmax=vmax),
    cbar_kwargs={"label": "mm in 6 hours"},
)
axis.add_patch(
    Rectangle(
        (-80, 36),
        14,
        9,
        fill=False,
        edgecolor="tab:red",
        linewidth=2,
        label="temperature comparison region",
    )
)
axis.legend(loc="lower left")
plt.title("FCN → Precipitation AFNO, initialized from real GFS")
print(f"Continental maximum: {float(precip_mm.max()):.1f} mm")
print(f"Finite pixels:    {float(np.isfinite(precip_mm).mean()):.1%}")

# %% [markdown]
# ## Where to go next
#
# - Verify both temperature forecasts against stations rather than choosing the
#   prettier map.
# - Compare the precipitation diagnostic with NOAA radar or gauge products.
# - Collocate the FCN wind fields with the real TEMPO/AQS case in notebook 05.
# - Record initialization time, lead time, units, and model/checkpoint versions
#   in every derived result.
