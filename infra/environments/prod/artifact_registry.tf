resource "google_artifact_registry_repository" "ml" {
  project       = var.project_id
  location      = var.region
  repository_id = "instacart-basket-prediction"
  format        = "DOCKER"
}