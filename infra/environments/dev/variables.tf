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

variable "project_roles" {
  description = "IAM roles at project level"
  type        = set(string)
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

variable "enable_nat_router" {
  type = bool
}

variable "enable_iap_ssh" {
  type = bool
}

variable "admin_user" {
  type = string
}

variable "subnet_cidr" {
  type = string
}

variable "master_machine_type" {
  description = "The type of worker machine used on dataproc master node"
  type        = string
  default     = "e2-standard-2"
}

variable "boot_disk_type" {
  description = "The type of disk used on dataproc cluster"
  type        = string
  default     = "pd-standard"
}

variable "master_boot_disk_size_gb" {
  description = "The size of disk used on dataproc master node"
  type        = number
  default     = 50
}

variable "worker_machine_type" {
  description = "The type of machine used on dataproc worker node"
  type        = string
  default     = "e2-stndard-2"
}

variable "worker_boot_disk_size_gb" {
  description = "The size of disk used on dataproc worker node"
  type        = number
  default     = 50
}

variable "image_version" {
  description = "The version of image used for dataproc"
  type        = string
  default     = "2.2-debian12"
}

variable "spark_shuffle_partitions" {
  type    = number
  default = 64
}

variable "spark_properties" {
  description = "Additional Dataproc software properties"
  type        = map(string)
  default     = {}
}

variable "optional_components" {
  description = "Optional Dataproc packages to install"
  type        = list(string)
  default     = []
}

variable "initialization_actions" {
  description = "Initialization scripts to run when cluster VMs start"

  type = list(object({
    script      = string
    timeout_sec = optional(number, 300)
  }))

  default = []
}

variable "worker_count" {
  description = "Number of primary Dataproc worker nodes"
  type        = number
  default     = 2

  validation {
    condition     = var.worker_count >= 2
    error_message = "worker_count must be at least 2 for a standard multi-node cluster"
  }
}

