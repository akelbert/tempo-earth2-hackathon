# Earth-2 Lab

The attendee container image and notebooks for pairing a NASA TEMPO retrieval
with an NVIDIA Earth2Studio forecast, built for the TEMPO × Earth-2 hackathon
(Harvard CfA, 14–16 September 2026).

The `terraform/` and `jupyterhub/` directories contain the environment-scoped
event deployment—see their own READMEs. No GCP project is hard-coded. Use a
dedicated project where possible; GPU defaults remain non-provisioning until
the benchmark, reservation, and budget gates pass.

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

`make demo-data` writes three Gaussian plumes over the northeast corridor,
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
| `TEMPO_DATA_URI` | Staged TEMPO Zarr store or directory |
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
