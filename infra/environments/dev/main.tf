resource "random_id" "bucket_suffix" {
  byte_length = 4
}

module "gcs_buckets" {
  project_id = var.project_id
  location   = var.region
  buckets    = var.buckets
  suffix     = random_id.bucket_suffix.hex
  prefix     = var.prefix

  source = "../../modules/gcs"

  depends_on = [google_project_service.required_apis]
}

module "iam" {
  source = "../../modules/iam"

  project_id         = var.project_id
  service_account_id = var.service_account_id
  display_name       = var.display_name
  user_email         = var.user_email

  project_roles = [
    "roles/dataproc.worker",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter"
  ]

  bucket_roles = {
    raw_admin = {
      bucket = module.gcs_buckets.bucket_names["raw"]
      role   = "roles/storage.objectAdmin"
    }

    bronze_admin = {
      bucket = module.gcs_buckets.bucket_names["bronze"]
      role   = "roles/storage.objectAdmin"
    }

    silver_admin = {
      bucket = module.gcs_buckets.bucket_names["silver"]
      role   = "roles/storage.objectAdmin"
    }

    gold_admin = {
      bucket = module.gcs_buckets.bucket_names["gold"]
      role   = "roles/storage.objectAdmin"
    }

    dataproc_staging_admin = {
      bucket = module.gcs_buckets.bucket_names["dataproc-staging"]
      role   = "roles/storage.objectAdmin"
    }
  }

  depends_on = [google_project_service.required_apis]
}