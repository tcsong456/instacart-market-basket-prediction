resource "google_service_account" "service_account" {
  project      = var.project_id
  account_id   = var.service_account_id
  display_name = var.display_name
}

resource "google_storage_bucket_iam_member" "bucket_roles" {
  for_each = var.bucket_roles

  bucket = each.value.bucket
  role   = each.value.role
  member = "serviceAccount:${google_service_account.service_account.email}"
}

resource "google_project_iam_member" "dataproc_workder" {
  for_each = var.project_roles

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.service_account.email}"
}

resource "google_project_iam_member" "user_dataproc_editor" {
  project = var.project_id
  role    = "roles/dataproc.editor"
  member  = "user:${var.admin_user}"
}

resource "google_service_account_iam_member" "user_act_as_dataproc_sa" {
  service_account_id = google_service_account.service_account.name
  role               = "roles/iam.serviceAccountUser"
  member             = "user:${var.admin_user}"
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