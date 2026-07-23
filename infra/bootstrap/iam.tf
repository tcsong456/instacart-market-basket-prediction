locals {
  terraform_iam_roles = toset([
    "roles/dataproc.editor",
    "roles/compute.networkAdmin",
    "roles/compute.securityAdmin",
    "roles/storage.admin",
    "roles/iam.serviceAccountAdmin",
    "roles/resourcemanager.projectIamAdmin",
    "roles/serviceusage.serviceUsageAdmin",
  ])
}

locals {
  workload_identity_pool_name = join("/", [
    "projects",
    data.google_project.current.number,
    "locations",
    "global",
    "workloadIdentityPools",
    google_iam_workload_identity_pool.github.workload_identity_pool_id,
  ])

  repository_principal_set = join("", [
    "principalSet://iam.googleapis.com/",
    local.workload_identity_pool_name,
    "/attribute.repository/",
    var.github_owner,
    "/",
    var.github_repository,
  ])

  main_branch_principal_set = join("", [
    "principalSet://iam.googleapis.com/",
    local.workload_identity_pool_name,
    "/attribute.ref/refs/heads/main",
  ])
}

resource "google_project_iam_member" "terraform_bootstrap_roles" {
  for_each = local.terraform_iam_roles

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.terraform_deployer.email}"
}

resource "google_project_iam_member" "planner_viewer" {
  project = var.project_id
  role    = "roles/viewer"

  member = "serviceAccount:${google_service_account.terraform_planner.email}"
}

resource "google_project_iam_member" "planner_service_usage" {
  project = var.project_id
  role    = "roles/serviceusage.serviceUsageConsumer"

  member = "serviceAccount:${google_service_account.terraform_planner.email}"
}

resource "google_storage_bucket_iam_member" "planner_state_reader" {
  bucket = google_storage_bucket.terraform_state.name
  role   = "roles/storage.objectViewer"

  member = "serviceAccount:${google_service_account.terraform_planner.email}"
}

resource "google_project_iam_custom_role" "terraform_storage_plan_reader" {
  project     = var.project_id
  role_id     = "terraformStoragePlanReader"
  title       = "Terraform Storage Plan Reader"
  description = "Allows Terraform plan to inspect Cloud Storage buckets and IAM policies."

  permissions = [
    "storage.buckets.get",
    "storage.buckets.list",
    "storage.buckets.getIamPolicy"
  ]
}

resource "google_project_iam_member" "planner_storage_reader" {
  project = var.project_id
  role    = google_project_iam_custom_role.terraform_storage_plan_reader.name
  member  = "serviceAccount:${google_service_account.terraform_planner.email}"
}

resource "google_service_account_iam_member" "github_plan" {
  service_account_id = google_service_account.terraform_planner.name

  role   = "roles/iam.workloadIdentityUser"
  member = local.repository_principal_set
}

resource "google_service_account_iam_member" "github_apply" {
  service_account_id = google_service_account.terraform_deployer.name

  role   = "roles/iam.workloadIdentityUser"
  member = local.main_branch_principal_set
}