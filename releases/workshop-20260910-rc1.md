# Notebook and model release candidate: workshop-20260910-rc1

Built and deployed September 10, 2026 in the designated Google Cloud project.
This is a release candidate for developer testing, not the final event freeze.

The Cloud Build source snapshot came from a dirty working tree based on Git
commit `d685944a566bd338bc9d36b6a7782c88ca9bd46a` (`main`). The final event image
must be rebuilt from a reviewed, committed source state.

## Immutable image and build

- Lab digest:
  `sha256:323f29a4ebd02191f7b510acc85973e4300938ebf761f1ea301bb0bbfeb2d89a`
- Pulled image size reported by GKE: 15,908,159,237 bytes (14.81 GiB)
- Successful Cloud Build: `73ccec98-6f65-409a-bed2-6224023e4cd8`
- Build duration: 26 minutes 49 seconds
- Earth2Studio requested version: 0.17.0
- Earth2Studio reported version: 0.17.0rc0
- PyTorch: 2.14.0+cu130; CUDA runtime: 13.0
- Installed extras: `fcn,dlwp,precip-afno,data,utils`
- Baked checkpoints: FCN, DLWP, and PrecipitationAFNO

The build context contained 70 source files totaling 1.0 MiB. Staged data,
Terraform state, certificates, private keys, credentials, local environments,
and Git metadata were excluded. One earlier submission failed before the image
build because the new third-party notice was omitted by `.dockerignore`; no
image was pushed by that attempt.

## Model terms and availability

`THIRD_PARTY_NOTICES.md` records the exact/default checkpoint sources and the
current NVIDIA governing-term links. The public NVIDIA NGC terms permit hosting
or distribution as part of a customer product subject to their requirements.
This engineering record is not a substitute for any Harvard-required legal or
procurement approval.

The catalog continues to label DLWP and PrecipitationAFNO as candidates in this
release candidate. Promote their catalog status in the final event release only
after the project owner accepts the terms review.

## Exact-image L4 gate

The digest above ran in a one-GPU GKE Job on the attendee node profile. The gate
required an NVIDIA L4, successful inference, finite diagnostic output, and zero
checkpoint-cache growth.

- Accelerator: NVIDIA L4, 22.03 GiB visible to PyTorch
- FCN: passed; 47.95 s cold process load, 3.20 s inference, 1.02 GiB peak GPU
- DLWP: passed; 1.87 s load, 1.40 s inference, 0.53 GiB peak GPU
- FCN to PrecipitationAFNO: passed; 5.00 s combined model load, 2.29 s
  inference, 1.36 GiB peak GPU
- Precipitation output: 100% finite, maximum 0.070994 m
- Runtime checkpoint-cache growth: 0.0 GiB for every workflow

The existing L4 was occupied, so the autoscaler successfully provisioned a
second node. A cold pull onto the new node took roughly eight minutes. Pulling
the changed layers onto the existing, already-warm GPU node took 1 minute 58
seconds.

## Notebook and data release

The attendee notebook order is:

1. `00_environment_check.ipynb`
2. `01_data_and_model_catalog.ipynb`
3. `02_tempo_earth2_toolkit_tour.ipynb`
4. `03_earth2_model_tasting_menu.ipynb`
5. `04_tempo_earth2_intro.ipynb`
6. `05_tempo_column_vs_surface.ipynb`
7. `06_smoke_event.ipynb`
8. `07_wetland_response.ipynb`
9. `08_scale_matters.ipynb`
10. `09_bring_your_own_site.ipynb`

The release includes collision-safe migration of old numbered workshop files
into each user's `.recovered` directory before the newly numbered notebooks are
seeded. Existing user edits and uploads are not overwritten.

The workshop bucket was synchronized with the real July 16 TEMPO smoke case,
NOAA HMS fire/smoke context, AirNow PM2.5, NASA HLS, NOAA CO-OPS, and updated
EPA AQS files. Canonical shared data remains read-only to attendee identities.

## Deployment validation

- JupyterHub Helm release: revision 6, chart 4.4.1, status `deployed`
- Hub, proxy, image pullers, and schedulers: Ready with zero restarts
- Existing user servers remained Running during the upgrade
- Both active GPU nodes pre-pulled the immutable lab digest
- Public `/hub/health`: HTTP 200 with successful certificate verification
- Public `/hub/login`: HTTP 200
- Public `/hub/signup`: HTTP 200; controlled registration remains enabled
- Current Hub image, TLS certificate, static IP, user allow list, persistent
  storage, and one-exclusive-L4-per-user policy were preserved

Already-running user servers retain the previous image until they are stopped
and restarted. Their persistent home volumes remain attached, so user-created
notebooks and uploads survive that restart.
