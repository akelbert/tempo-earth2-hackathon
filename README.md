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

`01_tempo_earth2_intro.ipynb` puts a real TEMPO retrieval and a real Earth-2
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

`02_tempo_column_vs_surface.ipynb` uses a separately staged, fully real matched
case: six TEMPO V04 scans from 31 May 2026 and same-day EPA AQS observations.
It intentionally does not invent meteorology when a real compatible source is
not present. The remaining short notebooks are possibility samplers; generated
fixtures keep them executable in CI while the corresponding real collections
are promoted through the catalog.

### Capability tiers

The machine-readable catalogs live beside the Python package as
`model_catalog.json` and `data_catalog.json`. Model tiers have operational
meaning:

- `guaranteed`: present in the released image and validated on the attendee L4;
- `candidate`: scientifically useful, but not promoted until the exact image
  passes the target-GPU benchmark and checkpoint terms are reviewed;
- `precomputed`: useful outputs can be supplied, but live inference is not
  promised on a 24 GB L4; and
- `conditional`: an upstream variable, credential, or compatibility issue must
  be resolved first.

FCN and persistence are currently guaranteed. DLWP and Precipitation AFNO
are the first candidates. StormCast, DLESyM, Aurora, and FCN3 remain visible to
attendees as project directions without pretending they fit the current pod.
Both candidates pass a one-step test in the exact candidate image on the local
8 GB RTX 2080 SUPER. DLWP peaked at 0.53 GB allocated GPU memory; the coupled
FCN-to-Precipitation-AFNO workflow peaked at 1.36 GB. These are encouraging
workstation results, not substitutes for the required L4 test or checkpoint
terms review, so neither candidate is advertised as live yet.

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
  forecast.py             Earth2Studio wrapper, grid handling, regridding
  advect.py               semi-Lagrangian advection and skill scoring
  plots.py                map and diagnostic plotting
  context.py              small CSV/Zarr context-data loaders and baselines
  catalog.py              query model/data capability tiers and resolved URIs
  data_catalog.json       real-source access and staging catalog
  model_catalog.json      validated/candidate/precomputed model catalog
  sources.py              bounded USGS and NOAA public-API helpers

scripts/
  stage_tempo.py          one-off: Earthdata → regional Zarr + manifest
  prefetch_models.py      warm the shared model cache; pre-bake forecasts
  build_notebooks.py      percent sources → .ipynb (supports cell tags)
  make_demo_data.py       synthetic TEMPO + forecast, for laptop testing
  make_teaching_data.py   deterministic context fixtures for all starter notebooks
  stage_aqs.py            EPA AQS hourly NO2/O3/PM2.5 teaching subset
  benchmark_models.py     target-L4 candidate promotion report
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
make publish-data BUCKET=gs://your-workshop-bucket
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

The contextual collection lives below `WORKSHOP_CONTEXT_DATA_URI`: air-quality
tables under `aqs/` and compact teaching inputs under `demo/`. Publishing syncs
both `data/tempo` and `data/context` into their corresponding bucket prefixes.
For broad AQS exploration, `stage-aqs-explore` writes monthly compressed
partitions; load one with `tempo_earth2.context.read_aqs_month("2026-05")`.

The deployed development bucket now also contains the validated real matched
case under `tempo/cases/2026-05-31/northeast.zarr`, its exact AQS table under
`context/aqs/2026-05-31.csv`, and monthly Northeast AQS partitions from
September 2025 through the currently published portion of June 2026. The
original `tempo/northeast.zarr` object set was not replaced.

Large archives are not copied wholesale. NOAA GFS/HRRR and bounded USGS/NOAA
APIs are accessed at source; NASA assets that require Earthdata credentials or
region-bound cloud access are staged into GCS by an operator; controlled
ForestGEO/investigator data require an explicit owner handoff and terms review.

### Build the attendee image

```bash
make build            # local, needs an amd64 Linux host with a GPU
make build-candidate  # non-deployable FCN/DLWP/precipitation validation image
make cloud-build-candidate RELEASE=rc1  # build/push it without changing Hub
make benchmark-candidates  # current GPU; L4 rerun required for promotion
make cloud-build      # Cloud Build, produces amd64 without emulation
make verify-image     # execute 00_environment_check.ipynb inside the image
make run              # run it locally with a GPU
```

To remove the event-day dependency on Hugging Face entirely, bake the checkpoint
into the image with `make build-baked` — after reviewing that model's license
for redistribution.

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
| `WORKSHOP_CONTEXT_DATA_URI` | Root containing staged AQS and teaching datasets |
| `EARTH2STUDIO_MODEL_CACHE` | Shared checkpoint cache, **outside** the user home |
| `EARTH2STUDIO_DATA_CACHE` | Writable data-source cache, inside the user home |
| `WORKSHOP_MODEL` | Default Earth2Studio model (`FCN`) |
| `WORKSHOP_RELEASE` | Release identifier; also gates notebook seeding |
| `WORKSHOP_PRECOMPUTED` | Optional reference forecasts for demonstrations, tests, and rapid iteration; not an availability fallback |
| `WORKSHOP_WORK_DIR` | Attendee working directory (`~/work`) |

The model cache is the one that matters most. Earth2Studio defaults it to
`~/.cache/earth2studio`, which on this deployment is a per-user persistent disk
— fifty attendees would each download the same multi-gigabyte checkpoint onto
their own volume. The image points it at `/opt/earth2/cache/models` instead.
