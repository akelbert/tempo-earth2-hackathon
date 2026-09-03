# Tainted so only attendee pods with the matching toleration (see
# jupyterhub/values-event.yaml.tmpl) land here - keeps system workloads off the
# expensive nodes, per the design doc's security requirements.

resource "google_container_node_pool" "gpu" {
  name     = "${local.resource_prefix}-gpu"
  cluster  = google_container_cluster.primary.id
  location = var.zone

  initial_node_count = var.gpu_initial_node_count

  autoscaling {
    min_node_count = var.gpu_min_node_count
    max_node_count = var.gpu_max_node_count
  }

  node_config {
    machine_type    = var.gpu_machine_type
    service_account = google_service_account.node.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    tags            = ["${local.resource_prefix}-node"]
    disk_size_gb    = var.node_disk_size_gb
    disk_type       = "pd-balanced"
    image_type      = "COS_CONTAINERD"

    gcfs_config {
      enabled = true
    }

    guest_accelerator {
      type  = var.gpu_type
      count = 1

      gpu_driver_installation_config {
        gpu_driver_version = var.gpu_driver_version
      }
    }

    labels = {
      workload = "jupyter-gpu"
    }

    taint {
      key    = "workload"
      value  = "jupyter-gpu"
      effect = "NO_SCHEDULE"
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }

    dynamic "reservation_affinity" {
      for_each = var.gpu_reservation_name == "" ? [] : [var.gpu_reservation_name]
      content {
        consume_reservation_type = "SPECIFIC_RESERVATION"
        key                      = "compute.googleapis.com/reservation-name"
        values                   = [reservation_affinity.value]
      }
    }
  }

  management {
    auto_repair = true
    # Event upgrades are frozen by the cluster maintenance exclusion and are
    # performed deliberately after compatibility testing.
    auto_upgrade = false
  }

  lifecycle {
    precondition {
      condition = (
        var.gpu_min_node_count <= var.gpu_initial_node_count &&
        var.gpu_initial_node_count <= var.gpu_max_node_count
      )
      error_message = "GPU node counts must satisfy min <= initial <= max."
    }
    precondition {
      condition     = var.environment != "event" || var.gpu_reservation_name != ""
      error_message = "The event environment requires gpu_reservation_name; do not rely on event-time on-demand capacity."
    }
  }

  depends_on = [
    google_project_iam_member.node_artifact_reader,
    google_project_iam_member.node_default,
  ]
}
