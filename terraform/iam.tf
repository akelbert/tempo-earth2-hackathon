# Two minimally privileged service accounts, per the design doc's security
# section: no broad project IAM roles, no service-account JSON keys, and the
# attendee workload uses Workload Identity Federation rather than the node's
# own identity.

resource "google_service_account" "node" {
  account_id   = "${local.resource_prefix}-node"
  display_name = "TEMPO x Earth-2 hackathon: GKE node service account"

  depends_on = [google_project_service.required["iam.googleapis.com"]]
}

resource "google_project_iam_member" "node_default" {
  project = var.project_id
  role    = "roles/container.defaultNodeServiceAccount"
  member  = "serviceAccount:${google_service_account.node.email}"
}

resource "google_project_iam_member" "node_artifact_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.node.email}"
}

# Bound to the Kubernetes service account the Helm chart's singleuser pods
# run as (values-event.yaml.tmpl: singleuser.serviceAccountName). No
# project-wide roles here - just object-level access to the specific buckets
# attendees need, granted alongside those buckets in storage.tf.
resource "google_service_account" "attendee" {
  account_id   = "${local.resource_prefix}-attendee"
  display_name = "TEMPO x Earth-2 hackathon: attendee workload identity"

  depends_on = [google_project_service.required["iam.googleapis.com"]]
}

resource "google_service_account_iam_member" "attendee_workload_identity" {
  service_account_id = google_service_account.attendee.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${var.k8s_namespace}/${var.k8s_service_account}]"
}

resource "kubernetes_namespace_v1" "jupyterhub" {
  metadata {
    name = var.k8s_namespace

    labels = {
      "pod-security.kubernetes.io/enforce" = "baseline"
      "pod-security.kubernetes.io/audit"   = "restricted"
      "pod-security.kubernetes.io/warn"    = "restricted"
    }
  }

  depends_on = [google_container_node_pool.system]
}

resource "kubernetes_service_account_v1" "attendee" {
  metadata {
    name      = var.k8s_service_account
    namespace = kubernetes_namespace_v1.jupyterhub.metadata[0].name

    annotations = {
      "iam.gke.io/gcp-service-account"          = google_service_account.attendee.email
      "iam.gke.io/return-principal-id-as-email" = "true"
    }
  }

  automount_service_account_token = false
}
