resource "google_dataproc_cluster" "instacart_cluster" {
  project = var.project_id
  name    = "${var.prefix}-${var.cluster_name}-${var.suffix}"
  region  = var.region

  cluster_config {
    staging_bucket = var.staging_bucket
    temp_bucket    = var.temp_bucket

    gce_cluster_config {
      zone            = "${var.region}-b"
      service_account = var.service_account_email
      service_account_scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
      ]

      subnetwork = var.subnetwork_uri

      internal_ip_only = true
      tags             = var.network_tags

      metadata = {
        block-project-ssh-keys = true
      }
    }

    master_config {
      num_instances = 1
      machine_type  = var.master_machine_type

      disk_config {
        boot_disk_type    = var.boot_disk_type
        boot_disk_size_gb = var.master_boot_disk_size_gb
      }
    }

    worker_config {
      num_instances = var.worker_count
      machine_type  = var.worker_machine_type

      disk_config {
        boot_disk_type    = var.boot_disk_type
        boot_disk_size_gb = var.worker_boot_disk_size_gb
      }
    }

    software_config {
      image_version = var.image_version

      override_properties = merge(
        {
          "spark:spark.sql.adaptive.enabled"                    = "true"
          "spark:spark.sql.adaptive.coalescePartitions.enabled" = "true"
          "spark:spark.sql.shuffle.partitions"                  = tostring(var.spark_shuffle_partitions)
        },
        var.spark_properties
      )

      optional_components = var.optional_components
    }

    dynamic "initialization_action" {
      for_each = var.initialization_actions

      content {
        script      = initialization_action.value.script
        timeout_sec = initialization_action.value.timeout_sec
      }
    }

    endpoint_config {
      enable_http_port_access = true
    }
  }

  lifecycle {
    prevent_destroy = false
  }
}