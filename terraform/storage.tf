# Canonical/optimized TEMPO data: read-only to attendees, per the design
# doc's "do not allow attendee pods to modify canonical inputs" requirement.
resource "google_storage_bucket" "tempo_data" {
  name          = "${var.project_id}-${local.resource_prefix}-tempo-data"
  location      = var.region
  storage_class = "STANDARD"
  force_destroy = false

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  retention_policy {
    retention_period = var.data_retention_seconds
    # Lock only the real event bucket. This is intentionally irreversible;
    # objects remain deletable after their individual retention period ends.
    is_locked = var.environment == "event"
  }

  labels = {
    event       = "tempo-earth2-hackathon"
    environment = var.environment
  }


  depends_on = [google_project_service.required["storage.googleapis.com"]]
}

resource "google_storage_bucket_iam_member" "tempo_data_attendee_reader" {
  bucket = google_storage_bucket.tempo_data.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.attendee.email}"
}

# Matches values-event.yaml.tmpl's singleuser.storage.dynamic.storageClass.
# reclaim_policy is Retain, not Delete: the design doc requires that
# destroying compute must not implicitly destroy user data.
resource "kubernetes_storage_class" "user_pd" {
  metadata {
    name = "workshop-user-pd-${var.environment}"

    labels = {
      environment = var.environment
    }
  }

  storage_provisioner    = "pd.csi.storage.gke.io"
  reclaim_policy         = "Retain"
  volume_binding_mode    = "WaitForFirstConsumer"
  allow_volume_expansion = true

  parameters = {
    type = "pd-balanced"
  }

  depends_on = [google_container_node_pool.system, kubernetes_namespace_v1.jupyterhub]
}
