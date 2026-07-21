variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "backend_bucket_name" {
  description = "The name of the bucket that stores backend state"
  type        = string
}

variable "deployer_service_account_id" {
  description = "Account ID for the Terraform deployment service account."
  type        = string
}

variable "planner_service_account_id" {
  description = "Account ID for the Terraform plan service account."
  type        = string
}

variable "workload_identity_pool_id" {
  description = "ID of the GitHub Workload Identity Pool."
  type        = string
}

variable "workload_identity_provider_id" {
  description = "ID of the GitHub Workload Identity Pool provider."
  type        = string
}

variable "github_owner" {
  description = "The owner of the GitHub"
  type        = string
}

variable "github_repository" {
  description = "The name of the github Repository that the terraform workflow is on"
  type        = string
}