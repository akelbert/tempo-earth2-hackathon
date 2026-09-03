variable "project_id" {
  description = "GCP project to deploy into. A dedicated event project is strongly preferred."
  type        = string

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{4,28}[a-z0-9]$", var.project_id))
    error_message = "project_id must be a valid 6-30 character Google Cloud project ID."
  }
}

variable "region" {
  description = "Region for regional resources (VPC, Artifact Registry, GCS). Matches the design doc's cost-estimate region."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Single zone for the cluster and its GPU capacity reservation."
  type        = string
  default     = "us-central1-a"
}

variable "name_prefix" {
  description = "Prefix applied to every resource this stack creates."
  type        = string
  default     = "tempo-earth2"

  validation {
    condition     = can(regex("^[a-z](?:[a-z0-9-]{0,12}[a-z0-9])?$", var.name_prefix))
    error_message = "name_prefix must be a lowercase Google Cloud name no longer than 14 characters."
  }
}

variable "environment" {
  description = "\"dev\" (rehearsal/sandbox, safe defaults, no GPU nodes unless raised explicitly) or \"event\" (the real deployment). Controls node pool sizing and deletion protection. The event-scale values should not be applied until the design doc's August 21/28 accelerator and budget gates have actually passed."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "event"], var.environment)
    error_message = "environment must be \"dev\" or \"event\"."
  }
}

variable "event_hostname" {
  description = "Public HTTPS hostname for this environment, without a scheme (for example living-earth-twin-hub.cfa.harvard.edu)."
  type        = string

  validation {
    condition = length(var.event_hostname) <= 253 && length(split(".", var.event_hostname)) > 1 && alltrue([
      for label in split(".", var.event_hostname) : can(regex("^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$", label))
    ])
    error_message = "event_hostname must be a lowercase fully qualified DNS name."
  }
}

variable "tls_certificate_name" {
  description = "Name of the global self-managed Google Cloud SSL certificate uploaded out-of-band. Certificate and private-key contents must never be passed to Terraform."
  type        = string

  validation {
    condition     = can(regex("^[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?$", var.tls_certificate_name))
    error_message = "tls_certificate_name must be a lowercase Google Cloud resource name."
  }
}

variable "master_authorized_networks" {
  description = "CIDR blocks allowed to reach the public GKE control-plane endpoint. Use operator/VPN /32 addresses; never 0.0.0.0/0."
  type = list(object({
    cidr_block   = string
    display_name = string
  }))

  validation {
    condition = length(var.master_authorized_networks) > 0 && alltrue([
      for network in var.master_authorized_networks : can(cidrhost(network.cidr_block, 0)) && network.cidr_block != "0.0.0.0/0"
    ])
    error_message = "Provide at least one restricted control-plane CIDR; 0.0.0.0/0 is forbidden."
  }
}

variable "k8s_namespace" {
  description = "Namespace the JupyterHub Helm release lives in."
  type        = string
  default     = "jupyterhub"
}

variable "k8s_service_account" {
  description = "Kubernetes service account the attendee pods run as; Helm obtains it from the kubernetes_service_account output."
  type        = string
  default     = "earth2-attendee"
}

# --- CPU system node pool (Hub, proxy, support services) --------------------

variable "cpu_node_count" {
  type    = number
  default = 2

  validation {
    condition     = var.cpu_node_count >= 2 && floor(var.cpu_node_count) == var.cpu_node_count
    error_message = "cpu_node_count must be an integer of at least 2."
  }
}

variable "cpu_machine_type" {
  type    = string
  default = "e2-standard-4"
}

# --- GPU attendee node pool ---------------------------------------------------

variable "gpu_initial_node_count" {
  description = "Initial GPU pool size. Keep at 0 in development; set to the pre-provisioned event capacity before rehearsal."
  type        = number
  default     = 0

  validation {
    condition     = var.gpu_initial_node_count >= 0 && floor(var.gpu_initial_node_count) == var.gpu_initial_node_count
    error_message = "gpu_initial_node_count must be a non-negative integer."
  }
}

variable "gpu_min_node_count" {
  description = "Minimum GPU nodes retained by the autoscaler. Set equal to event capacity during workshop hours."
  type        = number
  default     = 0

  validation {
    condition     = var.gpu_min_node_count >= 0 && floor(var.gpu_min_node_count) == var.gpu_min_node_count
    error_message = "gpu_min_node_count must be a non-negative integer."
  }
}

variable "gpu_max_node_count" {
  description = "Hard cost and capacity ceiling for GPU nodes."
  type        = number
  default     = 1

  validation {
    condition     = var.gpu_max_node_count >= 1 && floor(var.gpu_max_node_count) == var.gpu_max_node_count
    error_message = "gpu_max_node_count must be a positive integer."
  }
}

variable "gpu_machine_type" {
  description = "Design doc's preferred default: g2-standard-8 (1x L4, 8 vCPU, 32 GiB RAM). Revisit after the August 21 benchmark gate if the chosen model needs more VRAM than an L4 provides."
  type        = string
  default     = "g2-standard-8"
}

variable "gpu_type" {
  type    = string
  default = "nvidia-l4"
}

variable "gpu_driver_version" {
  description = "GKE managed GPU driver selection. DEFAULT is tested and frozen with the release; use LATEST only deliberately."
  type        = string
  default     = "DEFAULT"

  validation {
    condition     = contains(["DEFAULT", "LATEST"], var.gpu_driver_version)
    error_message = "gpu_driver_version must be DEFAULT or LATEST."
  }
}

variable "gpu_reservation_name" {
  description = "Name of the specific zonal Compute Engine reservation consumed by event GPU nodes. Empty is allowed only outside the event environment."
  type        = string
  default     = ""
}

variable "node_disk_size_gb" {
  description = "Boot disk size for the large scientific image, image cache, and measured ephemeral use."
  type        = number
  default     = 150

  validation {
    condition     = var.node_disk_size_gb >= 100 && floor(var.node_disk_size_gb) == var.node_disk_size_gb
    error_message = "node_disk_size_gb must be an integer of at least 100 GiB."
  }
}

variable "data_retention_seconds" {
  description = "Minimum retention period for canonical TEMPO objects. Defaults to 90 days."
  type        = number
  default     = 7776000
}

# --- Budget (monitoring.tf) ---------------------------------------------------

variable "billing_account_id" {
  description = "Billing account to attach a budget alert to. Empty skips the budget resource; an account administrator must grant the required billing role before setting it."
  type        = string
  default     = ""
}

variable "budget_ceiling_usd" {
  description = "Pre-credit budget ceiling. Design doc's expected range for the dedicated-L4 plan is $3,000-3,500; defaulting to the low end of that until a real ceiling is approved."
  type        = number
  default     = 3000

  validation {
    condition     = var.budget_ceiling_usd > 0
    error_message = "budget_ceiling_usd must be greater than zero."
  }
}

variable "alert_email" {
  description = "Operations email for event alert policies. Required when environment is event."
  type        = string
  default     = ""

  validation {
    condition     = var.alert_email == "" || can(regex("^[^@[:space:]]+@[^@[:space:]]+\\.[^@[:space:]]+$", var.alert_email))
    error_message = "alert_email must be empty or a plausible email address."
  }
}
