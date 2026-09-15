# Earth-2 Lab

The attendee container image and notebooks for pairing a NASA TEMPO retrieval
with an NVIDIA Earth2Studio forecast, built for the TEMPO × Earth-2 hackathon
(Harvard CfA, 14–16 September 2026).

The `terraform/` and `jupyterhub/` directories contain the environment-scoped
event deployment—see their own READMEs. No GCP project is hard-coded. Use a
dedicated project where possible; GPU defaults remain non-provisioning until
the benchmark, reservation, and budget gates pass.

---

## What the starter path does

`01_data_and_model_catalog.ipynb` is the landing notebook. It shows attendees
which capabilities are guaranteed, which models still need an L4 release test,
which heavier models are represented by precomputed outputs, and how to move
from a small example to the broader workshop collection or an authoritative
public archive.

`02_tempo_earth2_toolkit_tour.ipynb` explains the workshop-specific helper
modules, then uses Earth2Studio's native model, data-source, IO-backend, and
workflow interfaces for a minimal FCN run. `03_earth2_model_tasting_menu.ipynb`
uses the same native interface to compare FCN and DLWP and couple FCN to
Precipitation AFNO.

`04_tempo_earth2_intro.ipynb` begins the science path by putting a real TEMPO retrieval and a real Earth-2
forecast into the same array and produces a number:

1. Load a staged TEMPO tropospheric NO₂ scan and screen it for quality and
   cloud.
2. Run an Earth2Studio prognostic model (FourCastNet by default) from GFS
   initial conditions, snapping *down* to the most recent 6-hourly
   initialization so the model has not seen the observation.
3. Interpolate the forecast near-surface wind onto the TEMPO grid and overlay
   them.
4. Check the ventilation relationship — median NO₂ should fall as wind speed
   rises — as a sanity test before trusting anything else.
5. Advect the NO₂ field at scan *t₀* with the forecast wind and score it against
   scan *t₁*, using persistence as the baseline.
6. Offer ten follow-on projects, each of which attacks a stated limitation.

The advection is deliberately simple and the notebook says so at length: one
wind level stands in for a deep column, the wind is 6-hourly and 25 km against
an hourly 2 km observation, and there is no chemistry. Those are the hackathon,
not defects to hide.

`05_tempo_column_vs_surface.ipynb` uses a separately staged, fully real matched
case: six TEMPO V04 scans from 31 May 2026 and same-day EPA AQS observations,
then runs FCN and tests a baseline augmented with its live wind and temperature.
`06_smoke_event.ipynb` combines a documented July 2026 wildfire-smoke event
from NOAA HMS and AirNow with six real TEMPO scans and live FCN transport winds.
`07_wetland_response.ipynb` uses real NOAA Annapolis water levels and
astronomical tide predictions plus live FCN-to-Precipitation-AFNO output as
regional Chesapeake context. `08_scale_matters.ipynb` replaces the former
HLS-shaped fixture with a real seven-band, 30 m NASA HLS cutout over SERC and
compares its scale with live Earth-2 precipitation.
`09_bring_your_own_site.ipynb` remains a clearly labeled synthetic template
until an attendee or investigator supplies an approved site dataset.

Notebooks 03, 07, and 08 run the installed DLWP and Precipitation AFNO paths
directly through Earth2Studio. FCN, DLWP, and the coupled
FCN-to-Precipitation-AFNO workflow passed the exact-image L4 release gate with
zero checkpoint downloads. They do not silently fall back to CPU or synthetic
model output.

### Capability tiers

The machine-readable catalogs live beside the Python package as
`model_catalog.json` and `data_catalog.json`. Model tiers have operational
meaning:

- `guaranteed`: present in the released image and validated on the attendee L4;
- `reference`: visible for project planning, but not installed in the workshop
  image or promised on a 24 GB L4; and
- `conditional`: an upstream variable, credential, or compatibility issue must
  be resolved first.

FCN, DLWP, Precipitation AFNO, and persistence are guaranteed. On the exact
deployed image, DLWP peaked at 0.53 GB allocated L4 memory and the coupled
FCN-to-Precipitation-AFNO workflow peaked at 1.36 GB; neither downloaded a
checkpoint at runtime. StormCast, DLESyM, Aurora, and FCN3 remain reference-only
project directions and are not installed in the current image.

---

## Layout

```
terraform/
  network.tf, gke.tf, gpu_node_pool.tf, iam.tf, artifact_registry.tf,
  storage.tf, monitoring.tf     GCP infrastructure - see terraform/README.md

jupyterhub/
  values-event.yaml.tmpl   single digest-pinned GPU attendee profile
  render_values.py         validates and renders deployable Helm values

docker/
  Dockerfile              attendee image, built on nvcr.io/nvidia/pytorch
  requirements-lab.txt    pinned JupyterHub / Glue / geoscience stack
  entrypoint.sh           runs before jupyterhub-singleuser on every start
  bin/
    seed-workshop-notebooks     first-start seeding, never overwrites work
    restore-workshop-notebooks  clean copy of one notebook back
    reset-user-environment      undo a pip install that broke the environment

notebooks/
  src/*.py                percent-format sources (this is what you edit)
  *.ipynb                 rendered, shipped in the image (generated)

src/tempo_earth2/
  config.py               paths, env vars, environment reporting
  tempo.py                TEMPO loading, normalization, quality screening
  forecast.py             forecast convenience plus output/time/grid adapters
  advect.py               semi-Lagrangian advection and skill scoring
  plots.py                map and diagnostic plotting
  context.py              small CSV/Zarr context-data loaders and baselines
  catalog.py              query model/data capability tiers and resolved URIs
  data_catalog.json       real-source access and staging catalog
  model_catalog.json      installed/reference model availability catalog
  sources.py              bounded USGS and NOAA public-API helpers

scripts/
  stage_tempo.py          one-off: Earthdata → regional Zarr + manifest
  prefetch_models.py      warm the shared model cache; pre-bake forecasts
  build_notebooks.py      percent sources → .ipynb (supports cell tags)
  make_demo_data.py       synthetic TEMPO + forecast, for laptop testing
  make_teaching_data.py   deterministic context fixtures for all starter notebooks
  stage_aqs.py            EPA AQS hourly NO2/O3/PM2.5 teaching subset
  stage_coops.py          NOAA water-level/prediction case with provenance
  stage_smoke_case.py     NOAA HMS + preliminary AirNow event extract
  stage_hls.py            authenticated HLS COGs → bounded NetCDF cutout
  benchmark_models.py     exact-image target-L4 release validation report
  smoke_test.py           synthetic correctness check, no GPU needed
  test_notebook.py        execute a shipped .ipynb headlessly and assert on it
  _synthetic.py           TEMPO and forecast fixtures with a known answer
```

---

## Quick start

### Verify the analysis code (no Docker, no GPU, ~20 seconds)

```bash
make venv
make check
```

`make check` renders the notebooks, lints, and runs `smoke_test.py`, which
builds a synthetic TEMPO field containing a Gaussian plume, translates it by a
known distance, and asserts that advecting the first field with the
corresponding wind reproduces the second. That test exists to catch sign errors
and longitude-convention mistakes — the failure mode where a plume points the
wrong way and nobody notices until an attendee does.

### Stage TEMPO data (needs a NASA Earthdata login)

```bash
python -c "import earthaccess; earthaccess.login(persist=True)"
make stage-dry-run                    # see what it would download
make stage STAGE_DATE=2026-06-15
make publish-data BUCKET=gs://your-workshop-bucket
```

A full TEMPO L3 granule is about **850 MB**, and the notebook needs several
consecutive scans. Staging cuts that to a regional Zarr of tens of megabytes and
writes a `manifest.json` that the notebooks read to discover which scans exist,
so nobody has to guess a date.

### Prepare the starter-notebook data

```bash
make teaching-data
make stage-aqs STAGE_DATE=2026-06-15
make stage-aqs-explore AQS_START=2025-09-01 AQS_END=2026-06-01
make stage-coops-case
make stage-smoke-case
make stage-hls-case
make publish-data BUCKET=gs://your-workshop-bucket
make sync-real-case  # fetch the published May 31 TEMPO/AQS case for local use
```

`make teaching-data` builds small, deterministic context fixtures aligned with
the already staged TEMPO case. (`make demo-data` remains the all-synthetic path
for a separate laptop/test workspace and replaces its target TEMPO store.)
Every generated row and dataset is explicitly labeled synthetic; they are for
instruction and automated tests, not scientific evidence. `make stage-aqs`
replaces the matching air-quality fixture with public EPA AQS hourly
observations when that date has reached the annual bulk archive. AQS publication
lags real time, so a recent TEMPO case can legitimately have no matching rows.
The staging command fails instead of silently substituting a different date.

The contextual collection lives below `WORKSHOP_CONTEXT_DATA_URI`: validated
air-quality tables under `aqs/`, the real smoke event under `cases/`, the real
HLS cutout under `hls/`, coastal observations under `coops/`, and synthetic CI
fixtures under `demo/`. Publishing syncs both `data/tempo` and `data/context`
into their corresponding bucket prefixes.

For broad AQS exploration, `stage-aqs-explore` writes monthly compressed
partitions; load one with `tempo_earth2.context.read_aqs_month("2026-05")`.

The deployed development bucket now also contains the validated real matched
case under `tempo/cases/2026-05-31/northeast.zarr`, its exact AQS table under
`context/aqs/2026-05-31.csv`, and monthly Northeast AQS partitions from
September 2025 through the currently published portion of June 2026. The
original `tempo/northeast.zarr` object set was not replaced.

The July smoke case uses preliminary AirNow values because the regulatory AQS
annual bulk archive does not yet cover that event. Its manifest records that
limitation alongside the NOAA HMS and AirNow URLs. The HLS stager downloads
authenticated source COGs at preparation time, crops without spatial
resampling, applies the documented reflectance scale, and writes a small
analysis-ready NetCDF; attendees do not receive operator credentials.

Large archives are not copied wholesale. NOAA GFS/HRRR and bounded USGS/NOAA
APIs are accessed at source; NASA assets that require Earthdata credentials or
region-bound cloud access are staged into GCS by an operator; controlled
ForestGEO/investigator data require an explicit owner handoff and terms review.

### Build the attendee image

```bash
make build            # local, needs an amd64 Linux host with a GPU
make build-candidate  # FCN/DLWP/precipitation release-candidate image
make run-candidate    # local JupyterLab with those candidate extras
make cloud-build-candidate RELEASE=rc1  # build/push it without changing Hub
make benchmark-candidates  # current GPU; L4 rerun required for promotion
make render-l4-benchmark CANDIDATE_IMAGE_DIGEST='registry/image@sha256:...'
make cloud-build      # Cloud Build, produces amd64 without emulation
make verify-image     # execute 00_environment_check.ipynb inside the image
make run              # run it locally with a GPU
```

To remove the event-day dependency on Hugging Face entirely, bake the checkpoint
into the image with `make build-baked` — after reviewing that model's license
for redistribution.

`render-l4-benchmark` writes a one-GPU Job to
`jupyterhub/.generated/model-benchmark-job.yaml`. Rendering is safe and does
not contact the cluster. Before applying it, confirm that no demonstration or
attendee pod needs the available development GPU. The Job uses the exact
attendee resource envelope, refuses mutable image tags, runs FCN, DLWP, and the
coupled FCN-to-Precipitation-AFNO workflow, and leaves the JSON evidence in its
pod log. Candidate checkpoints are downloaded into an ephemeral cache in this
technical test; promotion into the event image additionally requires terms
review and checkpoint pre-baking.

`make demo-data DEMO_DATA=/path/to/disposable/demo` writes three Gaussian plumes over the northeast corridor,
drifting with a constant wind, in the same shape and units as real TEMPO L3.
It is not science — its value is that the answer is known, because the scans are
built by advecting the plumes with the forecast's own 10 m wind. The notebook
should score near **+0.98**. Anything else means the pipeline is broken, and you
learn that in ten seconds instead of after a 40-minute staging download.

---

## Configuration

Every path the notebooks use comes from an environment variable, so the same
image serves local development and the cluster.

| Variable | Purpose |
| --- | --- |
| `WORKSHOP_DATA_URI` | Root of the canonical workshop bucket; catalogs resolve dated collections below it |
| `TEMPO_DATA_URI` | Staged TEMPO Zarr store or directory |
| `WORKSHOP_CONTEXT_DATA_URI` | Root containing staged AQS, HMS/AirNow, HLS, coastal, and teaching datasets |
| `EARTH2STUDIO_MODEL_CACHE` | Shared checkpoint cache, **outside** the user home |
| `EARTH2STUDIO_DATA_CACHE` | Writable data-source cache, inside the user home |
| `WORKSHOP_MODEL` | Default Earth2Studio model (`FCN`) |
| `WORKSHOP_RELEASE` | Release identifier; also gates notebook seeding |
| `WORKSHOP_PRECOMPUTED` | Optional reference forecasts for demonstrations, tests, and rapid iteration; not an availability fallback |
| `WORKSHOP_WORK_DIR` | Attendee working directory (`~/work`) |

Set a new `WORKSHOP_RELEASE` whenever a deployed image changes the starter
notebooks. Seeding is intentionally skipped when a user's persistent home
already has the same release stamp. For the 00–09 renumbering, the seeder moves
old-numbered notebooks—including attendee edits—into
`~/work/.recovered/<timestamp>-renumbering/` before adding clean, newly numbered
copies. It never deletes or overwrites attendee work.

The model cache is the one that matters most. Earth2Studio defaults it to
`~/.cache/earth2studio`, which on this deployment is a per-user persistent disk
— fifty attendees would each download the same multi-gigabyte checkpoint onto
their own volume. The image points it at `/opt/earth2/cache/models` instead.
