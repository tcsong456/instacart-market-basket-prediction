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

resource "google_storage_bucket_iam_member" "remote_state_access" {
  bucket = google_storage_bucket.terraform_state.name
  role   = "roles/storage.objectAdmin"
  member = "user:${var.admin_user}"
}