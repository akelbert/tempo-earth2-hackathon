# %% [markdown]
# # Start here: data and Earth-2 model catalog
#
# The starter notebooks are intentionally small enough to finish during a
# workshop. They are **examples**, not the boundary of what you can investigate.
# This notebook shows three access levels:
#
# 1. reliable, analysis-ready example data staged next to JupyterHub;
# 2. broader workshop collections that you subset before loading; and
# 3. authoritative public archives/APIs that stay at their source.
#
# The model catalog makes a similarly important distinction: **guaranteed**
# models are installed and passed the event image's L4 test, **reference**
# models are project directions that are not installed, and **conditional**
# models first need an input or compatibility decision.

# %%
from tempo_earth2 import catalog
from tempo_earth2.config import WorkshopConfig

config = WorkshopConfig.from_env()

data_table = catalog.as_frame("data")
model_table = catalog.as_frame("models")

data_table

# %%
model_table

# %% [markdown]
# ## Open the configured TEMPO collection
#
# In JupyterHub, `TEMPO_DATA_URI` points at the Cloud Storage copy. The exact
# same code works against a local staged copy. Xarray opens Zarr lazily: this
# cell reads metadata, not the entire collection into memory.

# %%
import matplotlib.pyplot as plt

from tempo_earth2.tempo import apply_quality_mask, open_tempo_source

tempo = open_tempo_source(config.tempo_uri)
print(f"URI:       {config.tempo_uri}")
print(f"sizes:     {dict(tempo.sizes)}")
print(f"variables: {list(tempo.data_vars)}")
print(f"source:    {tempo.attrs.get('source', tempo.attrs.get('title', 'see manifest'))}")

if config.manifest:
    print(f"staged:    {config.manifest.get('created_utc', 'unknown')}")
    print(f"granules:  {len(config.manifest.get('granules', []))}")

# %%
field = apply_quality_mask(tempo)
if "time" in field.dims:
    field = field.isel(time=0)

field.plot(
    figsize=(10, 5),
    robust=True,
    cmap="magma",
    cbar_kwargs={"label": "tropospheric NO$_2$ (molecules cm$^{-2}$)"},
)
plt.title("First quality-screened TEMPO scan in the configured collection")
plt.tight_layout()

# %% [markdown]
# ## Find sources for your question
#
# Change the words below. A dataset can appear in several themes. The returned
# row tells you whether to use its `workshop_uri`, a bounded public request, or
# ask a data owner for controlled access.

# %%
MY_THEMES = ["wildfire", "smoke"]

matches = catalog.data_entries(MY_THEMES)
for item in matches:
    print(f"\n{item['title']} ({item['id']})")
    print(f"  access: {item['access_mode']} — {item['workshop_status']}")
    if item.get("workshop_uri"):
        print(f"  workshop: {item['workshop_uri']}")
    if item.get("direct_uri"):
        print(f"  archive:  {item['direct_uri']}")
    print(f"  source:   {item['source_url']}")
    print(f"  caution:  {item['notes']}")

# %% [markdown]
# ## Two bounded live-source examples
#
# `tempo_earth2.sources` includes small helpers for public USGS and NOAA APIs.
# They do not run automatically here because a notebook should still open if an
# external service has a transient outage. Remove `#` to make a narrow request.
#
# ```python
# from tempo_earth2.sources import read_noaa_coops, read_usgs_instantaneous_values
#
# stream = read_usgs_instantaneous_values(
#     "01100000", "2026-06-01", "2026-06-03", parameter_codes="00060"
# )
# tide = read_noaa_coops("8443970", "2026-06-01", "2026-06-03")
# ```
#
# For weather initialization, use Earth2Studio's `GFS()` datasource. For HRRR,
# select one run, forecast hour, variable group, and geographic window from
# `s3://noaa-hrrr-bdp-pds/`; never copy the archive recursively.

# %% [markdown]
# ## What “not limited to FCN” means here
#
# FCN is the default, but it is not the only live model. FCN and DLWP are
# installed prognostics, and PrecipitationAFNO is an installed diagnostic that
# consumes FCN output. The catalog also exposes model families relevant to
# storms, seasonal prediction, and foundation-model comparisons. Their `tier`
# tells you which paths are supported:
#
# - run live in your server (`guaranteed`);
# - treat as a project direction requiring separate hardware (`reference`); or
# - first resolve an input-compatibility issue (`conditional`).
#
# This avoids two bad surprises: discovering during the workshop that a 40–80
# GB model cannot fit a 24 GB L4, or having fifty users download checkpoints at
# once.

# %%
for item in catalog.model_entries():
    uses = ", ".join(item["science_uses"])
    print(f"{item['name']:<28} {item['tier']:<12} {uses}")

# %% [markdown]
# **Next reference:** open `02_tempo_earth2_toolkit_tour.ipynb` for a compact
# guide to the helper functions used throughout these notebooks and the basic
# live-model interfaces. Notebook 03 compares all installed live models.
