project_id = "instacart-basket"

region = "europe-west1"

buckets = {
  "raw"              = {}
  "bronze"           = {}
  "silver"           = {}
  "gold"             = {}
  "dataproc-staging" = { force_destroy = true }
  "dataproc-temp"    = { force_destroy = true }
}

service_account_id = "dataproc-etl-sa"

display_name = "ETL service account"

prefix = "instacart"

user_email = "user:congxisong@hotmail.com"

project_roles = [
  "roles/dataproc.worker",
  "roles/logging.logWriter",
  "roles/monitoring.metricWriter"
]

boot_disk_type = "pd-balanced"

worker_machine_type = "e2-standard-4"

master_machine_type = "e2-standard-4"

network_tags = ["dataproc-network"]

