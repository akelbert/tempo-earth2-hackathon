terraform {
  required_version = ">= 1.7"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.31"
    }
  }

  # State lives in its own bucket, bootstrapped once with gcloud before the
  # first `terraform init` - see README.md. Deliberately not configured here
  # with a hardcoded bucket name: dev/rehearsal and event deployments must
  # use separate state (see the design doc's "Deployment ownership" section),
  # so the bucket is passed via `-backend-config` per environment instead.
  backend "gcs" {}
}
