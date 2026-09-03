# Terraform: TEMPO × Earth-2 infrastructure

This stack creates the environment-specific VPC, private GKE Standard cluster,
system and GPU node pools, Workload Identity, Artifact Registry, canonical-data
bucket, global ingress IP, Cloud Build identity, and
billing budget. JupyterHub remains a separately invoked, pinned Helm release.

TLS uses the CfA-provided certificate for
`living-earth-twin-hub.cfa.harvard.edu`. Upload that certificate directly to
Google Cloud before applying Terraform; do not put its private key in this
directory, a `.tfvars` file, or Terraform state.

There is one attendee profile: a pod requesting one GPU on the GPU node pool.
The CPUs attached to that GPU node remain available for ordinary analysis, but
there is no separate CPU-fallback deployment.

## State bootstrap

Use a different state bucket for each environment. Restrict its IAM policy to
the Terraform operators; state is operationally sensitive.

```bash
gcloud storage buckets create gs://PROJECT-tempo-earth2-tfstate-dev \
  --project=PROJECT --location=us-central1 --uniform-bucket-level-access \
  --public-access-prevention
gcloud storage buckets update gs://PROJECT-tempo-earth2-tfstate-dev --versioning
```

Use a dedicated project for the event when possible. Resource names include the
environment, so dev and event can coexist if they must share a project.

## Apply

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Fill in the project, pre-shared certificate name, and your current public /32
# control-plane CIDR.

terraform init -backend-config="bucket=PROJECT-tempo-earth2-tfstate-dev"
terraform fmt -check -recursive
terraform validate
terraform plan -out=tfplan
terraform apply tfplan
```

The first apply enables required Google APIs. The operator needs Service Usage
Admin in addition to the permissions required to create the resources.

For event state, Terraform refuses to create the GPU pool without a specific
zonal reservation name. Set `gpu_initial_node_count`, `gpu_min_node_count`, and
`gpu_max_node_count` to the tested, authorized event capacity. Keeping min equal
to the event capacity prevents the autoscaler from releasing reserved workshop
nodes during event hours. Event plans also require `alert_email`; Terraform
creates a validated-HTTPS uptime check and an alert on GPU scheduling failures.

After apply:

```bash
terraform output -raw get_credentials_command
terraform output -raw ingress_ip
terraform output -raw tempo_data_uri
```

Before deploying the Ingress, protect and upload the institution-provided key
and full certificate chain from their secure out-of-repository location:

```bash
chmod 600 /secure/path/living-earth-twin-hub_cfa_harvard_edu.key
gcloud compute ssl-certificates create tempo-earth2-dev-tls-20260902 \
  --project=PROJECT --global \
  --certificate=/secure/path/living-earth-twin-hub_cfa_harvard_edu.pem \
  --private-key=/secure/path/living-earth-twin-hub_cfa_harvard_edu.key
```

The `.pem` must contain the leaf certificate followed by its intermediate
chain; it must not contain the private key. Set `tls_certificate_name` to the
created resource name. Google Cloud does not return the private key after
upload, so retain the administrator-provided bundle in approved secure storage.

Create the DNS A record for `living-earth-twin-hub.cfa.harvard.edu` at the
`ingress_ip` output. The Helm Ingress references the uploaded certificate by
resource name.

The person submitting Cloud Builds must have `iam.serviceAccounts.actAs` on the
output `cloud_build_service_account`. The dedicated build identity can read only
its source bucket and write only this environment's Artifact Registry repository.

## Teardown

Event cluster deletion protection is enabled. User PVs have a `Retain` reclaim
policy, and the event's canonical-data retention policy is locked (an
irreversible Google Cloud operation). Buckets have `force_destroy = false`.
Record the export/retention decision before disabling protection or destroying
compute. Removing the cluster does not constitute a user-data deletion
procedure.
