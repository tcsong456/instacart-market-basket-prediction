variable "project_id" {
  type = string
}

variable "display_name" {
  description = "Display name for the service account"
  type        = string
}

variable "service_account_id" {
  type = string
}

variable "bucket_roles" {
  description = "Bucket-level IAM roles assigned to the service account"
  type = map(object({
    bucket = string
    role   = string
  }))
}

variable "project_roles" {
  description = "Project-level IAM roles assigned to the service account"
  type        = set(string)
}

variable "user_email" {
  type = string
}