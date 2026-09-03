provider "google" {
  project = var.project_id
  region  = var.region
}

data "google_client_config" "default" {}
data "google_project" "current" {
  project_id = var.project_id
}

# Configured against the cluster this stack itself creates, so Terraform can
# also own the StorageClass attendee PVCs use (storage.tf). This is the
# standard "cluster + in-cluster resource in one stack" pattern; the caveat
# is that the very first `terraform apply` must create the cluster before it
# can create anything through this provider, so plans against an empty state
# will show the storage class as unknown until the cluster resource applies.
provider "kubernetes" {
  host                   = "https://${google_container_cluster.primary.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.primary.master_auth[0].cluster_ca_certificate)
}
