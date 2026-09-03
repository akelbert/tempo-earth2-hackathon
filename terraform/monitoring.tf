# Skipped entirely unless billing_account_id is set. The billing account
# A billing-account administrator must either create the budget separately or
# grant the operator Billing Account Costs Manager before enabling it here.
resource "google_billing_budget" "hackathon" {
  count = var.billing_account_id != "" ? 1 : 0

  billing_account = var.billing_account_id
  display_name    = local.resource_prefix

  budget_filter {
    projects = ["projects/${data.google_project.current.number}"]
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.budget_ceiling_usd)
    }
  }

  # Design doc: notify at 50/75/90/100% of the approved pre-credit ceiling.
  # These are alerts, not spending caps - the actual control is the explicit
  # node-pool min/max bounds in gpu_node_pool.tf.
  threshold_rules {
    threshold_percent = 0.5
  }
  threshold_rules {
    threshold_percent = 0.75
  }
  threshold_rules {
    threshold_percent = 0.9
  }
  threshold_rules {
    threshold_percent = 1.0
  }

  depends_on = [google_project_service.required["billingbudgets.googleapis.com"]]
}

resource "google_monitoring_notification_channel" "operations_email" {
  count = var.alert_email == "" ? 0 : 1

  display_name = "${local.resource_prefix} operations"
  type         = "email"
  labels = {
    email_address = var.alert_email
  }

  depends_on = [google_project_service.required["monitoring.googleapis.com"]]
}

# Exercise the exact public TLS path attendees use, including certificate
# validation. The check may fail while DNS/certificate provisioning converges;
# event readiness requires it to recover before the URL is distributed.
resource "google_monitoring_uptime_check_config" "hub_https" {
  count = var.environment == "event" ? 1 : 0

  display_name     = "${local.resource_prefix} Hub HTTPS"
  timeout          = "10s"
  period           = "60s"
  selected_regions = ["USA"]

  monitored_resource {
    type = "uptime_url"
    labels = {
      project_id = var.project_id
      host       = var.event_hostname
    }
  }

  http_check {
    request_method = "GET"
    path           = "/hub/health"
    port           = 443
    use_ssl        = true
    validate_ssl   = true
  }

  depends_on = [google_project_service.required["monitoring.googleapis.com"]]
}

resource "google_monitoring_alert_policy" "hub_https" {
  count = var.environment == "event" ? 1 : 0

  display_name = "${local.resource_prefix}: public Hub HTTPS unavailable"
  combiner     = "OR"

  conditions {
    display_name = "HTTPS health check failing"

    condition_threshold {
      filter = join(" AND ", [
        "metric.type=\"monitoring.googleapis.com/uptime_check/check_passed\"",
        "resource.type=\"uptime_url\"",
        "metric.label.check_id=\"${google_monitoring_uptime_check_config.hub_https[0].uptime_check_id}\"",
      ])
      comparison      = "COMPARISON_LT"
      threshold_value = 1
      duration        = "300s"

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_FRACTION_TRUE"
      }

      trigger {
        count = 1
      }
    }
  }

  notification_channels = google_monitoring_notification_channel.operations_email[*].name

  alert_strategy {
    auto_close = "1800s"
  }

  lifecycle {
    precondition {
      condition     = var.alert_email != ""
      error_message = "The event environment requires alert_email for operational alerts."
    }
  }
}

# GKE scheduler events are the most direct signal that attendee pods cannot
# obtain a GPU. Keep this metric in both environments for rehearsals.
resource "google_logging_metric" "gpu_scheduling_failures" {
  name        = "${replace(local.resource_prefix, "-", "_")}_gpu_scheduling_failures"
  description = "FailedScheduling events involving GPU capacity or placement."
  filter = join(" AND ", [
    "resource.type=\"k8s_pod\"",
    "resource.labels.cluster_name=\"${google_container_cluster.primary.name}\"",
    "jsonPayload.reason=\"FailedScheduling\"",
    "jsonPayload.message=~\"(?i)(nvidia.com/gpu|reservation|node affinity|untolerated taint)\"",
  ])

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }

  depends_on = [google_project_service.required["logging.googleapis.com"]]
}

resource "google_monitoring_alert_policy" "gpu_scheduling_failures" {
  count = var.environment == "event" ? 1 : 0

  display_name = "${local.resource_prefix}: attendee GPU scheduling failures"
  combiner     = "OR"

  conditions {
    display_name = "GPU FailedScheduling event observed"

    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.gpu_scheduling_failures.name}\" AND resource.type=\"k8s_pod\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      duration        = "0s"

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_DELTA"
        cross_series_reducer = "REDUCE_SUM"
      }

      trigger {
        count = 1
      }
    }
  }

  notification_channels = google_monitoring_notification_channel.operations_email[*].name

  alert_strategy {
    auto_close = "1800s"
  }

  lifecycle {
    precondition {
      condition     = var.alert_email != ""
      error_message = "The event environment requires alert_email for operational alerts."
    }
  }
}
