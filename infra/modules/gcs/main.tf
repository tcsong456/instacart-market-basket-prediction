resource "google_storage_bucket" "bucket" {
  for_each = var.buckets

  project  = var.project_id
  name     = "${var.prefix}-${each.key}-${var.suffix}"
  location = var.location

  force_destroy = each.value.force_destroy
  storage_class = each.value.storage_class

  versioning {
    enabled = each.value.versioning
  }

  uniform_bucket_level_access = true
}