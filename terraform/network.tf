# Dedicated VPC for the hackathon cluster - deliberately not the project's
# `default` network, so this stack's IP ranges and firewall rules can never
# collide with unrelated project workloads.

resource "google_compute_network" "vpc" {
  name                    = "${local.resource_prefix}-vpc"
  auto_create_subnetworks = false
  description             = "TEMPO x Earth-2 hackathon - isolated from other workloads in this project."

  depends_on = [google_project_service.required["compute.googleapis.com"]]
}

resource "google_compute_subnetwork" "subnet" {
  name          = "${local.resource_prefix}-subnet"
  ip_cidr_range = "10.60.0.0/20"
  region        = var.region
  network       = google_compute_network.vpc.id

  secondary_ip_range {
    range_name    = "${local.resource_prefix}-pods"
    ip_cidr_range = "10.61.0.0/16"
  }

  secondary_ip_range {
    range_name    = "${local.resource_prefix}-services"
    ip_cidr_range = "10.62.0.0/20"
  }

  private_ip_google_access = true
}

resource "google_compute_firewall" "allow_internal" {
  name    = "${local.resource_prefix}-allow-internal"
  network = google_compute_network.vpc.id

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }
  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }
  allow {
    protocol = "icmp"
  }

  source_ranges = [
    google_compute_subnetwork.subnet.ip_cidr_range,
    google_compute_subnetwork.subnet.secondary_ip_range[0].ip_cidr_range,
    google_compute_subnetwork.subnet.secondary_ip_range[1].ip_cidr_range,
  ]
}

# Google's documented health-check source ranges - required for the
# load balancer in front of the Hub/proxy to reach node ports.
resource "google_compute_firewall" "allow_health_checks" {
  name    = "${local.resource_prefix}-allow-health-checks"
  network = google_compute_network.vpc.id

  allow {
    protocol = "tcp"
  }

  source_ranges = ["35.191.0.0/16", "130.211.0.0/22"]
  target_tags   = ["${local.resource_prefix}-node"]
}

# Private nodes use Cloud NAT for package/data access without public node IPs.
resource "google_compute_router" "nat" {
  name    = "${local.resource_prefix}-router"
  network = google_compute_network.vpc.id
  region  = var.region
}

resource "google_compute_router_nat" "nat" {
  name                               = "${local.resource_prefix}-nat"
  router                             = google_compute_router.nat.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "LIST_OF_SUBNETWORKS"

  subnetwork {
    name                    = google_compute_subnetwork.subnet.id
    source_ip_ranges_to_nat = ["ALL_IP_RANGES"]
  }
}

# External GKE Ingress uses a global address. DNS and the institution-provided
# pre-shared certificate are deliberately coordinated outside this stack so
# private-key material never enters Terraform configuration or state.
resource "google_compute_global_address" "ingress_ip" {
  name        = "${local.resource_prefix}-ingress-ip"
  description = "Global public IP for the JupyterHub HTTPS Ingress."

  depends_on = [google_project_service.required["compute.googleapis.com"]]
}
