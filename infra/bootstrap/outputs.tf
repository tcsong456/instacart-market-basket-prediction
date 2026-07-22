output "state_bucket_name" {
  description = "The name of the remote terraform state bucket"
  value       = google_storage_bucket.terraform_state.name
}

output "terraform_deployer_email" {
  value = google_service_account.terraform_deployer.email
}

output "terraform_planner_email" {
  description = "Terraform planner service account email."
  value       = google_service_account.terraform_planner.email
}

output "workload_identity_pool_name" {
  description = "Full resource name of the Workload Identity Pool."
  value       = google_iam_workload_identity_pool.github.name
}

output "workload_identity_provider_name" {
  description = "Full resource name the Workload Identity provider."
  value       = google_iam_workload_identity_pool_provider.github.name
}