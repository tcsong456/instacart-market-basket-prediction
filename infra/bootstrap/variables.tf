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

variable "admin_user" {
  type = string
}