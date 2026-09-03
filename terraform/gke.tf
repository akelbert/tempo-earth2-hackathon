# GKE Standard, not Autopilot - the design doc's "Why GKE Standard" section
# is explicit about needing separate tainted node pools, explicit GPU
# placement, and node-pool autoscaling bounds that Autopilot doesn't expose
# the same way.

resource "google_container_cluster" "primary" {
  name     = local.resource_prefix
  location = var.zone

  network    = google_compute_network.vpc.id
  subnetwork = google_compute_subnetwork.subnet.id

  networking_mode   = "VPC_NATIVE"
  datapath_provider = "ADVANCED_DATAPATH"
  ip_allocation_policy {
    cluster_secondary_range_name  = google_compute_subnetwork.subnet.secondary_ip_range[0].range_name
    services_secondary_range_name = google_compute_subnetwork.subnet.secondary_ip_range[1].range_name
  }

  # Node pools are managed as separate google_container_node_pool resources
  # (below, and in gpu_node_pool.tf) so each can be sized, tainted, and
  # autoscaled independently - never the cluster's own default pool.
  remove_default_node_pool = true
  initial_node_count       = 1

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }

  master_authorized_networks_config {
    dynamic "cidr_blocks" {
      for_each = var.master_authorized_networks
      content {
        cidr_block   = cidr_blocks.value.cidr_block
        display_name = cidr_blocks.value.display_name
      }
    }
  }

  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }

  monitoring_config {
    enable_components = [
      "SYSTEM_COMPONENTS",
      "APISERVER",
      "SCHEDULER",
      "CONTROLLER_MANAGER",
      "STORAGE",
      "HPA",
      "POD",
      "DAEMONSET",
      "DEPLOYMENT",
      "STATEFULSET",
      "CADVISOR",
      "KUBELET",
    ]

    managed_prometheus {
      enabled = true
    }
  }

  cost_management_config {
    enabled = true
  }

  resource_labels = {
    event       = "tempo-earth2-hackathon"
    environment = var.environment
  }

  release_channel {
    channel = "REGULAR"
  }

  # Placeholder window; the design doc calls for constraining disruptive
  # automatic maintenance during event hours once the September dates are
  # locked in - revisit this before the release-candidate freeze.
  maintenance_policy {
    daily_maintenance_window {
      start_time = "09:00"
    }

    dynamic "maintenance_exclusion" {
      for_each = var.environment == "event" ? [1] : []
      content {
        exclusion_name = "tempo-earth2-event-freeze"
        start_time     = "2026-09-13T00:00:00Z"
        end_time       = "2026-09-17T12:00:00Z"

        exclusion_options {
          scope = "NO_UPGRADES"
        }
      }
    }
  }

  # Only protect the real event cluster from accidental `terraform destroy`;
  # a dev/rehearsal stack should stay easy to tear down and rebuild.
  deletion_protection = var.environment == "event"

  depends_on = [
    google_project_service.required["compute.googleapis.com"],
    google_project_service.required["container.googleapis.com"],
    google_compute_router_nat.nat,
  ]
}

resource "google_container_node_pool" "system" {
  name     = "${local.resource_prefix}-system"
  cluster  = google_container_cluster.primary.id
  location = var.zone

  node_count = var.cpu_node_count

  node_config {
    machine_type    = var.cpu_machine_type
    service_account = google_service_account.node.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    tags            = ["${local.resource_prefix}-node"]
    disk_size_gb    = var.node_disk_size_gb
    disk_type       = "pd-balanced"
    image_type      = "COS_CONTAINERD"

    # Enable image streaming on each managed node pool. Event nodes are still
    # pre-warmed; streaming is not a substitute for rehearsal.
    gcfs_config {
      enabled = true
    }

    labels = {
      workload = "system"
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  depends_on = [
    google_project_iam_member.node_artifact_reader,
    google_project_iam_member.node_default,
  ]
}
