data "terraform_remote_state" "bootstrap" {
  backend = "gcs"

  config = {
    bucket = "instacart-terraform-state-3c0d312b"
    prefix = "bootstrap"
  }
}

resource "google_service_account" "terraform_etl" {
  project      = var.project_id
  account_id   = var.etl_service_account_id
  display_name = "Data ETL service account"
}

resource "google_storage_bucket_iam_member" "dataproc_staging_admin" {
  bucket = var.dataproc_staging_bucket_name
  role   = "roles/storage.admin"
  member = "serviceAccount:${google_service_account.terraform_etl.email}"
}

resource "google_storage_bucket_iam_member" "dataproc_temp_admin" {
  bucket = var.dataproc_temp_bucket_name
  role   = "roles/storage.admin"
  member = "serviceAccount:${google_service_account.terraform_etl.email}"
}

resource "google_storage_bucket_iam_member" "bucket_roles" {
  for_each = var.bucket_roles

  bucket = each.value.bucket
  role   = each.value.role
  member = "serviceAccount:${google_service_account.terraform_etl.email}"
}

resource "google_project_iam_member" "dataproc_worker" {
  for_each = var.project_roles

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.terraform_etl.email}"
}

resource "google_service_account_iam_member" "deployer_use_etl_runtime" {
  service_account_id = google_service_account.terraform_etl.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${data.terraform_remote_state.bootstrap.outputs.terraform_deployer_email}"
}

resource "google_project_iam_member" "user_dataproc_editor" {
  project = var.project_id
  role    = "roles/dataproc.editor"
  member  = "user:${var.admin_user}"
}

resource "google_service_account_iam_member" "user_dataproc_submit" {
  service_account_id = google_service_account.terraform_etl.name
  role = "roles/iam.serviceAccountUser"
  member = "user:${var.admin_user}"
}

resource "google_project_iam_member" "iap_tunnel_accessor" {
  count = var.enable_iap_ssh ? 1 : 0

  project = var.project_id
  role    = "roles/iap.tunnelResourceAccessor"
  member  = "user:${var.admin_user}"
}

resource "google_project_iam_member" "os_login" {
  count = var.enable_iap_ssh ? 1 : 0

  project = var.project_id
  role    = "roles/compute.osAdminLogin"
  member  = "user:${var.admin_user}"
}