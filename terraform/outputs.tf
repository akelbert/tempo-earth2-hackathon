output "cluster_name" {
  value = google_container_cluster.primary.name
}

output "cluster_endpoint" {
  value     = google_container_cluster.primary.endpoint
  sensitive = true
}

output "get_credentials_command" {
  value = "gcloud container clusters get-credentials ${google_container_cluster.primary.name} --zone ${var.zone} --project ${var.project_id}"
}

output "artifact_registry_repository" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.earth2.repository_id}"
}

output "artifact_registry_repository_id" {
  value = google_artifact_registry_repository.earth2.repository_id
}

output "tempo_data_bucket" {
  value = google_storage_bucket.tempo_data.name
}

output "tempo_data_uri" {
  value = "gs://${google_storage_bucket.tempo_data.name}/tempo/northeast.zarr"
}

output "workshop_data_uri" {
  value = "gs://${google_storage_bucket.tempo_data.name}"
}

output "context_data_uri" {
  value = "gs://${google_storage_bucket.tempo_data.name}/context"
}

output "attendee_service_account" {
  value = google_service_account.attendee.email
}

output "ingress_ip" {
  value = google_compute_global_address.ingress_ip.address
}

output "ingress_ip_name" {
  value = google_compute_global_address.ingress_ip.name
}

output "tls_certificate_name" {
  value = var.tls_certificate_name
}

output "event_hostname" {
  value = var.event_hostname
}

output "user_storage_class" {
  value = kubernetes_storage_class.user_pd.metadata[0].name
}

output "kubernetes_service_account" {
  value = kubernetes_service_account_v1.attendee.metadata[0].name
}

output "cloud_build_service_account" {
  value = google_service_account.cloud_build.name
}

output "cloud_build_source_bucket" {
  value = google_storage_bucket.cloud_build_source.name
}
