# Model availability normalization: workshop-20260910-rc2

Built, target-L4 validated, and deployed September 10, 2026 in the designated
Google Cloud project. This is a release candidate for workshop-team testing,
not the final committed event freeze.

The Cloud Build source snapshot came from a dirty working tree based on Git
commit `d685944a566bd338bc9d36b6a7782c88ca9bd46a` (`main`). The final event image
must be rebuilt from a reviewed, committed source state.

## Immutable image and build

- Lab digest:
  `sha256:183254b97f9d4f340bffc99cbb3284d386bb752a6cb28b926f880240e4d2f042`
- Pulled image size reported by GKE: 15,908,166,862 bytes (14.81 GiB)
- Successful Cloud Build: `57374a50-5bc6-4e07-94f5-d796aa1f0b2a`
- Build duration: 18 minutes 54 seconds
- Earth2Studio requested version: 0.17.0
- Earth2Studio resolved commit: `d027d61277229081f9e11c74638260f2ade67631`
- Earth2Studio reported version: 0.17.0rc0
- PyTorch: 2.14.0+cu130; CUDA runtime: 13.0
- Installed extras: `fcn,dlwp,precip-afno,data,utils`
- Baked checkpoints: FCN, DLWP, and PrecipitationAFNO

The uploaded build context contained 70 source files totaling 1.0 MiB. Staged
data, Terraform state, certificates, private keys, credentials, local
environments, build outputs, and Git metadata were excluded.

An earlier attempt, `4c471901-d1cd-40f5-a049-e35d1e36c4f3`, failed before
image export when the public Hugging Face API returned HTTP 429 while FCN was
being prefetched. No image was pushed by that attempt. The prefetcher now
honors registry-provided retry delays for transient 429 and 5xx responses,
while permanent errors continue to fail the build immediately.

## Normalized model availability

The catalog, helper API, notebooks, image contents, and exact-image gate now
describe the same event-day capability:

- FCN and DLWP are installed, live, L4-validated prognostic models.
- PrecipitationAFNO is an installed, live, L4-validated diagnostic coupled to
  FCN.
- Persistence is a guaranteed scientific baseline, not a neural model.
- StormCast, DLESyM, Aurora, and FCN3 are reference-only project directions and
  are explicitly not installed.
- SolarRadiationAFNO remains conditional because the guaranteed workflow does
  not supply all of its required inputs.

`tempo_earth2.forecast.SUPPORTED_PROGNOSTICS` and
`SUPPORTED_DIAGNOSTICS` expose the installed inventory without presenting a
diagnostic as a prognostic. `SUPPORTED_MODELS` remains only as a backwards-
compatible prognostic alias.

The model-facing cells in notebooks 02, 03, 07, and 08 use Earth2Studio's
native model, data, IO, and workflow APIs. The removed `candidate_models`
wrapper is no longer part of the attendee package. Workshop helpers remain for
configuration, data access, quality control, time/grid alignment, diagnostics,
and plotting.

## Exact-image L4 gate

The immutable digest above ran in the uniquely named GKE Job
`earth2-model-benchmark-workshop-20260910-rc2`. The gate required an NVIDIA L4,
successful inference, finite diagnostic output, and zero checkpoint-cache
growth.

- Accelerator: NVIDIA L4, 22.03 GiB visible to PyTorch
- FCN: passed; 48.22 s load, 3.30 s inference, 1.02 GiB peak GPU
- DLWP: passed; 1.36 s load, 1.50 s inference, 0.53 GiB peak GPU
- FCN to PrecipitationAFNO: passed; 1.60 s FCN load, 3.43 s diagnostic
  load, 2.27 s inference, 1.36 GiB peak GPU
- Precipitation output: shape 1 x 2 x 720 x 1440, 100% finite, maximum
  0.070994146 m
- Runtime checkpoint-cache growth: 0.0 GiB for every workflow

An existing developer server occupied the first L4. The benchmark requested its
own exclusive GPU and triggered the GPU pool from one node to two without
preemption or sharing. The cold pull took 7 minutes 27 seconds. The temporary
second node later scaled down after the Job completed.

## Local and deployment validation

- Notebook sources rendered reproducibly.
- Ruff and JSON validation passed.
- All 33 synthetic analysis smoke checks passed.
- All ten notebooks executed without error in the CPU-safe test runner; it
  skipped cells tagged `requires-gpu`, while the same native FCN, DLWP, and
  precipitation workflows were exercised by the exact-image L4 gate.
- JupyterHub Helm release: revision 7, chart 4.4.1, status `deployed`.
- Hub, proxy, image puller, and schedulers: Ready with zero restarts.
- The continuous image puller uses the immutable RC2 digest.
- Public `/hub/health`, `/hub/login`, and `/hub/signup`: HTTPS 200 with
  successful certificate verification.
- Controlled signup remains enabled and the six-user allow list is unchanged.
- Static IP, hostname, TLS certificate, Hub image, persistent storage, culling,
  and one-exclusive-L4-per-user policy were preserved.

The already-running developer server remained on the RC1 digest and was not
restarted. It received RC2 after a normal stop/start; its persistent home
volume and user-created content survived that restart. All newly started
servers used RC2.
