resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "google_storage_bucket" "terraform_state" {
  project  = var.project_id
  name     = "${var.backend_bucket_name}-${random_id.bucket_suffix.hex}"
  location = var.region

  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  force_destroy = false

  versioning {
    enabled = true
  }

  lifecycle {
    prevent_destroy = true
  }

  labels = {
    purpose = "terraform-state"
  }
}

resource "google_service_account" "terraform_deployer" {
  project = var.project_id

  account_id   = var.deployer_service_account_id
  display_name = "Terraform deployer"
  description  = "Service account used by GitHub Actions to run Terraform"

  depends_on = [google_project_service.bootstrap]
}

resource "google_service_account" "terraform_planner" {
  project = var.project_id

  account_id   = var.planner_service_account_id
  display_name = "Terraform planner"
  description  = "Service account used by GitHub Actions to plan Terraform"

  depends_on = [google_project_service.bootstrap]
}