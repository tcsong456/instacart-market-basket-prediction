data "terraform_remote_state" "bootstrap" {
  backend = "gcs"

  config = {
    bucket = "instacart-terraform-state-3c0d312b"
    prefix = "bootstrap"
  }
}

resource "google_service_account" "training_image_builder" {
  project = var.project_id

  account_id   = "training-image-builder"
  display_name = "Training Image Builder"
}

resource "google_service_account_iam_member" "github_image_builder" {
  service_account_id = google_service_account.training_image_builder.name

  role = "roles/iam.workloadIdentityUser"

  member = data.terraform_remote_state.bootstrap.outputs.main_branch_principal_set
}

resource "google_artifact_registry_repository_iam_member" "image_builder" {
  project    = var.project_id
  location   = var.artifact_registry_location
  repository = var.artifact_registry_repository_id

  role   = "roles/artifactregistry.writer"
  member = "serviceAccount:${google_service_account.training_image_builder.email}"
}

resource "google_service_account" "runpod_artifact_reader" {
  project = var.project_id

  account_id   = "runpod-artifact-reader"
  display_name = "RunPod Artifact Registry Reader"
}

resource "google_artifact_registry_repository_iam_member" "runpod_reader" {
  project    = var.project_id
  location   = var.artifact_registry_location
  repository = var.artifact_registry_repository_id

  role = "roles/artifactregistry.reader"

  member = "serviceAccount:${google_service_account.runpod_artifact_reader.email}"
}

resource "google_service_account" "training_runtime" {
  project      = var.project_id
  account_id   = "training-runtime"
  display_name = "Training Runtime Service Account"
}

resource "google_storage_bucket_iam_member" "training_runtime_read_data" {
  bucket = var.bucket_names["gold"]

  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.training_runtime.email}"
}

resource "google_storage_bucket_iam_member" "training_runtime_write_runs" {
  bucket = var.bucket_names["runs"]

  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.training_runtime.email}"
}

resource "google_service_account" "training_runner" {
  project      = var.project_id
  account_id   = "training-runner"
  display_name = "GitHub Training Runner"
}

resource "google_storage_bucket_iam_member" "training_runner_runs_reader" {
  bucket = var.bucket_names["runs"]
  role   = "roles/storage.objectViewer"

  member = "serviceAccount:${google_service_account.training_runner.email}"
}

resource "google_service_account_iam_member" "training_runner_build" {
  service_account_id = google_service_account.training_runner.name

  role = "roles/iam.workloadIdentityUser"

  member = data.terraform_remote_state.bootstrap.outputs.repository_principal_set
}

resource "google_service_account" "terraform_etl" {
  project      = var.project_id
  account_id   = var.etl_service_account_id
  display_name = "Data ETL service account"
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

resource "google_service_account_iam_member" "admin_use_etl_runtime" {
  service_account_id = google_service_account.terraform_etl.name
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