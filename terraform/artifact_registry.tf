# The Makefile reads this environment-specific repository ID from Terraform
# state so development and event builds cannot silently share image tags.

resource "google_artifact_registry_repository" "earth2" {
  location      = var.region
  repository_id = local.resource_prefix
  format        = "DOCKER"
  description   = "Attendee image releases for the TEMPO x Earth-2 hackathon."

  docker_config {
    immutable_tags = var.environment == "event"
  }


  labels = {
    event       = "tempo-earth2-hackathon"
    environment = var.environment
  }

  depends_on = [google_project_service.required["artifactregistry.googleapis.com"]]
}
