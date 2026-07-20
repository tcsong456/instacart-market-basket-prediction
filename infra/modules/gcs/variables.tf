variable "project_id" {
  description = "Google cloud project ID"
  type        = string
}

variable "location" {
  description = "Bucket location"
  type        = string
}

variable "buckets" {
  description = "Map of buckets to create"

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

variable "suffix" {
  description = "gcs bucket name suffix"
  type        = string
}