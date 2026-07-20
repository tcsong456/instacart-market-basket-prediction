terraform {
  backend "gcs" {
    bucket = "instacart-terraform-state-3c0d312b"
    prefix = "environments/dev"
  }
}