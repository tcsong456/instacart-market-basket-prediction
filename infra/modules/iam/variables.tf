variable "project_id" {
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

variable "admin_user" {
  type = string
}

variable "enable_iap_ssh" {
  description = "Wether to use iap ssh get access to VMs"
  type        = bool
}

variable "etl_service_account_id" {
  description = "The name of the etl service account"
  type        = string
}

variable "dataproc_staging_bucket_name" {
  description = "Name of the Dataproc staging bucket"
  type        = string
}

variable "dataproc_temp_bucket_name" {
  description = "Name of the Dataproc temporary bucket"
  type        = string
}