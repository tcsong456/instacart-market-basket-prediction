project_id = "instacart-basket"

region = "europe-west1"

buckets = {
  "raw"              = {}
  "bronze"           = {}
  "silver"           = {}
  "gold"             = {}
  "dataproc-staging" = { force_destroy = true }
}

service_account_id = "dataproc-etl-sa"

display_name = "ETL service account"

prefix = "instacart"

user_email = "user:congxisong@hotmail.com"

