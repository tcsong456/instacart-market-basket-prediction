variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "buckets" {
  type = map(object({
    storage_class = optional(string, "STANDARD")
    versioning    = optional(bool, false)
    force_destroy = optional(bool, false)
  }))
}

variable "prefix" {
  description = "gcs bucket name prefix"
  type        = string
}

variable "service_account_id" {
  type = string
}

variable "display_name" {
  type = string
}

variable "user_email" {
  type = string
}