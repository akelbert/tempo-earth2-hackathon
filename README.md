# Earth-2 Lab

The attendee container image and introductory notebook for the TEMPO × Earth-2
hackathon (Harvard CfA, 14–16 September 2026).

This is the implementation of the container and notebook tracks described in
[`tempo_earth2_jupyterhub_gke_design.md`](tempo_earth2_jupyterhub_gke_design.md).
Terraform and the JupyterHub Helm release are separate and not in this
repository yet.

---

## What the notebook does

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

**This addresses the "TEMPO as a diagnostic dataset alongside model output"
branch of the scientific scope gate.** If the science team decides TEMPO should
instead be a model *input*, most of this repository still applies — the
container, staging, caching and recovery paths do not change — but §5 of the
notebook would be replaced.

---

## Layout

```
docker/
  Dockerfile              attendee image, built on nvcr.io/nvidia/pytorch
  Dockerfile.cpu          CPU development variant; builds on Apple Silicon
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

scripts/
  stage_tempo.py          one-off: Earthdata → regional Zarr + manifest
  prefetch_models.py      warm the shared model cache; pre-bake forecasts
  build_notebooks.py      percent sources → .ipynb (supports cell tags)
  make_demo_data.py       synthetic TEMPO + forecast, for laptop testing
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

### Build the attendee image

```bash
make build            # local, needs an amd64 Linux host with a GPU
make cloud-build      # Cloud Build, produces amd64 without emulation
make verify-image     # execute 00_environment_check.ipynb inside the image
make run              # run it locally with a GPU
```

To remove the event-day dependency on Hugging Face entirely, bake the checkpoint
into the image with `make build-baked` — after reviewing that model's license
for redistribution.

### On an Apple Silicon Mac

The attendee image is not useful on a Mac: its base is ~20 GB of CUDA with no
device to execute it, and cross-building it for amd64 under emulation takes
hours. Use the CPU development variant, which builds natively on arm64:

```bash
make demo-data        # synthetic TEMPO scans, no Earthdata login needed
make build-cpu        # docker/Dockerfile.cpu, python:3.13-slim base
make verify-cpu       # execute both notebooks inside the image, headless
make verify-shell     # recovery commands resolve in a login shell
make lab-cpu          # open JupyterLab at http://localhost:8888/lab
make lab-cpu-reset    # discard the work volume, i.e. become a new attendee
```

Your edits persist in a named Docker volume across `make lab-cpu` runs, which is
what makes the seeding behavior testable: a second start must not overwrite an
edited notebook, and `restore-workshop-notebooks` must move your version aside
rather than delete it.

`make demo-data` writes three Gaussian plumes over the northeast corridor,
drifting with a constant wind, in the same shape and units as real TEMPO L3.
It is not science — its value is that the answer is known, because the scans are
built by advecting the plumes with the forecast's own 10 m wind. The notebook
should score near **+0.98**. Anything else means the pipeline is broken, and you
learn that in ten seconds instead of after a 40-minute staging download.

The CPU image tells you whether the notebooks run, whether the widgets render,
whether the Glue viewers behave in a real browser, and whether the seeding and
recovery commands work. It tells you **nothing** about GPU memory, inference
time, CUDA compatibility, or the NGC dependency resolution.

---

## Configuration

Every path the notebooks use comes from an environment variable, so the same
image serves local development and the cluster.

| Variable | Purpose |
| --- | --- |
| `TEMPO_DATA_URI` | Staged TEMPO Zarr store or directory |
| `EARTH2STUDIO_MODEL_CACHE` | Shared checkpoint cache, **outside** the user home |
| `EARTH2STUDIO_DATA_CACHE` | Writable data-source cache, inside the user home |
| `WORKSHOP_MODEL` | Default Earth2Studio model (`FCN`) |
| `WORKSHOP_RELEASE` | Release identifier; also gates notebook seeding |
| `WORKSHOP_PRECOMPUTED` | Pre-baked forecasts for the CPU fallback path |
| `WORKSHOP_WORK_DIR` | Attendee working directory (`~/work`) |

The model cache is the one that matters most. Earth2Studio defaults it to
`~/.cache/earth2studio`, which on this deployment is a per-user persistent disk
— fifty attendees would each download the same multi-gigabyte checkpoint onto
their own volume. The image points it at `/opt/earth2/cache/models` instead.

---

## What has and has not been verified

Verified on this machine, all via `make check`:

- The advection, regridding, grid-normalization, time-selection and plotting
  code, against synthetic data with a known answer (28 checks in
  `smoke_test.py`). The load-bearing one: a Gaussian plume is translated by a
  known distance, and advecting the earlier field with the corresponding wind
  must land the peak on the later one to within a grid cell.
- **Both shipped notebooks execute end to end**, headless, through nbclient
  (`test_notebook.py`). The introductory notebook is run against synthetic TEMPO
  scans built by advecting plumes with the synthetic forecast's own 10 m wind,
  and it must recover that wind: it scores **+0.985** at 10 m, degrading to
  +0.798 and +0.600 for the scaled 100 m and 850 hPa winds. That confirms the
  wind-level selector takes effect and the grids genuinely correspond.
- The Dockerfile passes `docker buildx build --check` with no warnings, and both
  base images (`nvcr.io/nvidia/pytorch:26.04-py3`, `ghcr.io/astral-sh/uv`)
  resolve and are publicly pullable.
- Package versions and the Earth2Studio extras list are current as of the pins
  in `requirements-lab.txt`.
- The TEMPO collection identifiers: `TEMPO_NO2_L3` **V04** is the ongoing
  collection (`C3685896708-LARC_CLOUD`); V03 ended 2025-09-16.

Additionally verified inside the built CPU image on an Apple Silicon Mac: both
notebooks execute, JupyterLab serves, the container runs as non-root `jovyan`,
notebook seeding does not overwrite an edited notebook on restart,
`restore-workshop-notebooks` moves the attendee's version aside rather than
deleting it, and `reset-user-environment` correctly reports "nothing to reset"
on a clean environment.

That pass caught one real bug, now fixed in both Dockerfiles and guarded by
`make verify-shell`: a JupyterLab terminal starts a **login** shell, and
`/etc/profile` rebuilds `PATH` from scratch, discarding the image's `ENV PATH`.
Every recovery command was "command not found" at precisely the moment an
attendee would have been told to run one. The fix is a `/etc/profile.d` snippet.

Two cells are tagged `optional` and excluded from headless runs: the ipywidgets
slider and the Glue viewers. Both need a live frontend to mean anything, and the
Glue cell stalls a headless nbclient kernel even though its body raises
`ImportError` in isolation. Nothing downstream uses either. `--with-optional`
reproduces the stall; **both still need testing in a real browser**, which is
the venue-network validation the design document already calls for.

**Not** verified, because it needs an amd64 GPU host:

- That the image actually builds. The Earth2Studio and lab-stack installs are
  separate layers with a version-assertion step between them, so a resolver
  conflict fails the build rather than silently downgrading torch — but that
  assertion has not been exercised.
- Peak GPU memory, model load time, warm inference time, or cache size for
  FourCastNet. **These are the numbers the accelerator decision depends on**, and
  they are what `00_environment_check.ipynb` plus a first notebook run will
  produce.
- Whether Earth2Studio can consume a *read-only* shared model cache without
  attempting a write. The image sets the cache read-only for attendees; if that
  fails, the fallback is a node-local writable cache populated before pods start.

**Verified in a real browser, one confirmed defect found:** glue-jupyter's
linked map/scatter views render and link correctly, but the map viewer's own
selection tool crashes (`bqplot-image-gl`/`bqplot-gl` interaction-binding bug;
no newer release exists to pin against). The scatter viewer's selection tool
works and drives the same linked highlight on the map, so the notebook now
directs attendees to select there instead. Glue is a required part of the
guided path (see the design document), not an optional accessibility extra, so
this workaround - not a matplotlib-only fallback - is the accepted mitigation
until an upstream fix ships.

---

## Open decisions this does not settle

These belong to the scientific scope gate, not to the container:

1. Whether TEMPO is a diagnostic dataset (what this notebook assumes), a model
   input, or a linked visualization.
2. Which model and pinned checkpoint. `FCN` is chosen here because it needs only
   `nvidia-physicsnemo`, is served from Hugging Face rather than NGC, and
   produces 10 m / 100 m / 850 hPa winds. It is a defensible default, not a
   decision.
3. The event date to stage. `stage_tempo.py` takes `--date`; pick one with good
   coverage over the chosen region and few clouds, and stage several candidates.
4. Who owns the reference notebooks after this one.
