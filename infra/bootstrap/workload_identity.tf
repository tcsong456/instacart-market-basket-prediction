data "google_project" "current" {
  project_id = var.project_id
}

resource "google_iam_workload_identity_pool" "github" {
  project = var.project_id

  workload_identity_pool_id = var.workload_identity_pool_id
  display_name              = "GitHub Actions"
  description               = "Federated identities from approved GitHub Actions workflows"

  depends_on = [google_project_service.bootstrap]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  project = var.project_id

  workload_identity_pool_id = google_iam_workload_identity_pool.github.workload_identity_pool_id

  workload_identity_pool_provider_id = var.workload_identity_provider_id
  display_name                       = "GitHub Actions OIDC"
  description                        = "Trust provider for the configured GitHub repository"

  attribute_mapping = {
    "google.subject"             = "assertion.sub"
    "attribute.actor"            = "assertion.actor"
    "attribute.repository"       = "assertion.repository"
    "attribute.repository_owner" = "assertion.repository_owner"
    "attribute.ref"              = "assertion.ref"
  }

  attribute_condition = join(" && ", [
    "assertion.repository_owner == '${var.github_owner}'",
    "assertion.repository == '${var.github_owner}/${var.github_repository}'"
  ])

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}