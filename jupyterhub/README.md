# JupyterHub: TEMPO × Earth-2 hackathon

This directory contains one supported attendee profile: one GPU-requesting
JupyterLab pod per active attendee. Ordinary CPU analysis runs on the CPUs that
come with the GPU node. There is no separate CPU-fallback deployment; loss of
GPU capacity is handled as a platform incident.

The Zero to JupyterHub chart is pinned to **4.4.1**. Its Hub application version
is 5.5.1 and matches the custom Hub and single-user images.

## 1. Apply Terraform and fetch credentials

```bash
terraform -chdir=../terraform output -raw get_credentials_command
# Run the printed command.
```

Terraform owns the namespace, attendee Kubernetes ServiceAccount and its
Workload Identity annotation. Helm must not be installed before those exist.

## 2. Build digest-pinned images

`make cloud-build` builds and pushes both images. Its final log lines contain
full Artifact Registry references ending in `@sha256:...`; use those exact
references below. The event repository uses immutable tags, but Helm still
deploys by digest.

## 3. Render values

The template is not a deployable values file. Rendering refuses unresolved
placeholders, mutable image tags, unsafe usernames, an empty allow-list, or a
non-GCS workshop data path.

```bash
export HUB_IMAGE_REFERENCE='REGION-docker.pkg.dev/PROJECT/REPOSITORY/earth2-hub@sha256:...'
export LAB_IMAGE_REFERENCE='REGION-docker.pkg.dev/PROJECT/REPOSITORY/earth2-lab@sha256:...'
export EVENT_HOSTNAME='living-earth-twin-hub.cfa.harvard.edu'
export INGRESS_IP_NAME="$(terraform -chdir=../terraform output -raw ingress_ip_name)"
export TLS_CERTIFICATE_NAME="$(terraform -chdir=../terraform output -raw tls_certificate_name)"
export STORAGE_CLASS_NAME="$(terraform -chdir=../terraform output -raw user_storage_class)"
export KUBERNETES_SERVICE_ACCOUNT="$(terraform -chdir=../terraform output -raw kubernetes_service_account)"
export WORKSHOP_DATA_URI="$(terraform -chdir=../terraform output -raw workshop_data_uri)"
export TEMPO_DATA_URI="$(terraform -chdir=../terraform output -raw tempo_data_uri)"
export WORKSHOP_CONTEXT_DATA_URI="$(terraform -chdir=../terraform output -raw context_data_uri)"
export WORKSHOP_RELEASE='2026-09-rc1'
export WORKSHOP_ADMIN_USERNAME='actual-admin-username'
export WORKSHOP_ALLOWED_USERS='actual-admin-username,attendee1,attendee2'
export ENABLE_SIGNUP='true'

make -C .. helm-template
```

Inspect `jupyterhub/.generated/values-event.yaml`, then deploy with:

```bash
make -C .. deploy-hub
```

`deploy-hub` templates chart 4.4.1 first and uses Helm's rollback-on-failure
behavior.

For release validation, `render_benchmark_job.py` creates an ignored,
digest-pinned, uniquely named one-GPU Job manifest. It only renders the manifest; an operator
must confirm spare GPU capacity before applying it so release testing cannot
displace a demo or attendee server.

## Authentication workflow

NativeAuthenticator does not provide an administrator-facing bulk account
creation screen. Use this controlled workflow:

1. Create DNS and wait for the certificate and final HTTPS hostname to work,
   but do not distribute that hostname to attendees yet.
2. Sign up the configured administrator through the final HTTPS path. A name
   listed in `admin_users` is authorized automatically when it is created.
3. During a supervised window, attendees sign up using identifiers present in
   `WORKSHOP_ALLOWED_USERS`; the administrator verifies and authorizes them.
4. After registration, render and deploy again with `ENABLE_SIGNUP=false`.
5. Test rejected usernames, password reset, restart, and a new browser before
   distributing the URL.

JupyterHub 5.5 includes the apparent client IP in anonymous XSRF tokens. Google
Front Ends can select different proxy IPs for the signup-page GET and its POST,
which otherwise causes `XSRF cookie does not match POST argument`. The event
values normalize Google's `35.191.0.0/16` and `130.211.0.0/22` proxy ranges,
including the IPv4-mapped IPv6 forms GKE currently presents to the Hub. Do not
remove that setting while this deployment uses the GKE external Application
Load Balancer.

The allow-list must include the administrator. Do not distribute an installation
that still contains an example username or has signup enabled after registration.

## HTTPS and DNS

Terraform creates a global address. The CfA-provided certificate is uploaded
out-of-band as a global self-managed Google Cloud SSL certificate so its private
key never enters Git or Terraform state. The Helm Ingress attaches that
pre-shared certificate and the Terraform-managed address to an external GKE
Application Load Balancer and disables plain HTTP. Create the DNS A record for
`living-earth-twin-hub.cfa.harvard.edu` using:

```bash
terraform -chdir=../terraform output -raw ingress_ip
```

Do not treat the deployment as ready until the certificate is attached to the
target HTTPS proxy and the final hostname passes browser and WebSocket tests.
The supplied leaf certificate expires March 19, 2027; record its institutional
renewal owner even though that date is after the workshop.

Kubernetes prints a deprecation warning for the
`kubernetes.io/ingress.class: gce` annotation. This warning is expected: GKE
Ingress still requires that annotation and ignores `spec.ingressClassName` for
controller selection. Do not replace it merely to silence the generic warning.

## Workload Identity and network isolation

The attendee KSA impersonates a bucket-scoped Google service account. The
chart's privileged metadata-blocking init container is disabled; GKE Dataplane
V2 instead enforces a NetworkPolicy that allows the controlled metadata server
while denying arbitrary private-network access from attendee pods.

Verify this from a non-admin attendee pod: TEMPO reads must succeed, writes to
the canonical bucket must fail, another attendee pod must be unreachable, and
the Compute Engine node metadata identity must not be exposed.

## Incident boundary

If the attendee pod remains running but CUDA or model inference fails, users can
continue CPU analysis and save work. If GPU nodes cannot run GPU-requesting pods,
servers remain Pending and JupyterHub is unavailable to those users. Operators
restore reserved GPU capacity using the event runbook; switching to a second
CPU deployment is not a supported recovery action.
