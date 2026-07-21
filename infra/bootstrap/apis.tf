locals {
  bootstrap_apis = toset([
    "iam.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iamcredentials.googleapis.com",
    "sts.googleapis.com",
    "serviceusage.googleapis.com"
  ])
}

resource "google_project_service" "bootstrap" {
  for_each = local.bootstrap_apis

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}