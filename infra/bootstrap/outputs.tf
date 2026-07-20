output "state_bucket_name" {
  description = "The name of the remote terraform state bucket"
  value       = google_storage_bucket.terraform_state.name
}