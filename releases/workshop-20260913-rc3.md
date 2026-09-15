# Event release: workshop-20260913-rc3

Built, target-L4 validated, and deployed September 13, 2026 in the designated
Google Cloud project. This release adds NASA Harmony access and freezes the
workshop deployment on 32 reserved, exclusive NVIDIA L4 GPUs.

The Cloud Build source snapshot came from a dirty working tree. Before any
later rebuild, commit and review the intended source changes; attendee roster
data and generated Helm values must remain outside version control.

## Immutable image and build

- Lab digest:
  `sha256:25926e345fdd48d88618079810b95f760b53044df06e05ac5c04a47cbff72302`
- Successful Cloud Build: `13eb9b9f-e0e2-43d4-8fa6-d346cf21ccb8`
- Build duration: 19 minutes 8 seconds
- Earth2Studio reported version: 0.17.0rc0
- PyTorch: 2.14.0+cu130; CUDA runtime: 13.0
- Baked models: FCN, DLWP, and PrecipitationAFNO
- Added NASA client: `harmony-py==1.6.0`

The uploaded build context contained 71 source files totaling 1.0 MiB. Staged
data, Terraform state, generated Helm values, certificates, private keys,
credentials, local environments, build outputs, and Git metadata were
excluded.

## Exact-image L4 gate

The immutable digest above ran in GKE Job
`earth2-harmony-benchmark-20260913-rc1`. The gate required an NVIDIA L4,
successful inference, finite diagnostic output, and zero runtime checkpoint
downloads.

- FCN: passed; 47.95 s load, 3.10 s inference, 1.02 GiB peak GPU
- DLWP: passed; 1.39 s load, 1.50 s inference, 0.53 GiB peak GPU
- FCN to PrecipitationAFNO: passed; 1.64 s FCN load, 3.39 s diagnostic
  load, 2.34 s inference, 1.36 GiB peak GPU
- Precipitation output: shape 1 x 2 x 720 x 1440, 100% finite, maximum
  0.070994146 m
- Runtime checkpoint-cache growth: 0.0 GiB for every workflow

The container build also imported `harmony` successfully as part of its
package and ABI validation layer.

## Event capacity and registration

- Specific reservation: `tempo-earth2-workshop-20260914`
- Zone and shape: `us-central1-a`, 32 `g2-standard-8` VMs, one L4 each
- Reservation status after provisioning: READY, 32 assured, 32 in use
- GPU node pool: fixed at 32 nodes for the event; all 32 Ready
- One exclusive L4 is requested by every attendee server; GPU sharing remains
  disabled.
- The continuous image puller is Ready on all 32 GPU nodes.
- JupyterHub Helm release: revision 12, chart 4.4.1, status `deployed`.
- Registration permits 83 usernames: roster email addresses, three documented
  missing-email fallbacks, ten guest names, and retained existing accounts.
  The names themselves remain only in the ignored generated values file.
- Native signup is enabled. New accounts still require administrator approval.
- Idle attendee servers are culled after 90 minutes without Hub activity; the
  culler checks every 10 minutes, so reclamation can occur between 90 and 100
  minutes after the last recorded activity.
- Public `/hub/signup`: HTTPS 200 after deployment.

The live stack deliberately retains its rehearsed `dev` resource names; an
environment-label change would replace most infrastructure immediately before
the workshop. A non-empty reservation name now enables both cluster deletion
protection and the event maintenance exclusion through September 17, 2026.

## End-of-event warning

The reservation and its 32 consuming nodes are billable now. After the final
workshop session, first remove the fixed 32-node minimum and release the
consuming GPU nodes, then delete the specific reservation. Do not attempt to
delete the reservation while its capacity is still in use.
