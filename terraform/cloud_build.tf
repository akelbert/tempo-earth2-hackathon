resource "google_service_account" "cloud_build" {
  account_id   = "${local.resource_prefix}-build"
  display_name = "TEMPO x Earth-2 hackathon: Cloud Build runtime"

  depends_on = [google_project_service.required["iam.googleapis.com"]]
}

resource "google_artifact_registry_repository_iam_member" "cloud_build_writer" {
  project    = var.project_id
  location   = google_artifact_registry_repository.earth2.location
  repository = google_artifact_registry_repository.earth2.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.cloud_build.email}"
}

resource "google_project_iam_member" "cloud_build_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.cloud_build.email}"
}

resource "google_storage_bucket" "cloud_build_source" {
  name                        = "${var.project_id}-${local.resource_prefix}-build-source"
  location                    = var.region
  storage_class               = "STANDARD"
  force_destroy               = false
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  lifecycle_rule {
    condition {
      age = 14
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    event       = "tempo-earth2-hackathon"
    environment = var.environment
  }

  depends_on = [google_project_service.required["storage.googleapis.com"]]
}

resource "google_storage_bucket_iam_member" "cloud_build_source_reader" {
  bucket = google_storage_bucket.cloud_build_source.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.cloud_build.email}"
}
