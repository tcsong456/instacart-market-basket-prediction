variable "suffix" {
  description = "Suffix for dataproc name"
  type        = string
}

variable "prefix" {
  description = "Prefix for dataproc name"
  type        = string
}

variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "cluster_name" {
  type = string
}

variable "staging_bucket" {
  description = "staging bucket used to load up files for dataproc cluster"
  type        = string
}

variable "temp_bucket" {
  description = "scratch space for dataproc cluster"
  type        = string
}

variable "service_account_email" {
  type = string
}

variable "worker_count" {
  description = "Number of primary Dataproc worker nodes"
  type        = number
}

variable "master_machine_type" {
  description = "The type of worker machine used on dataproc master node"
  type        = string
}

variable "boot_disk_type" {
  description = "The type of disk used on dataproc cluster"
  type        = string
}

variable "master_boot_disk_size_gb" {
  description = "The size of disk used on dataproc master node"
  type        = number
}

variable "worker_machine_type" {
  description = "The type of machine used on dataproc worker node"
  type        = string
}

variable "worker_boot_disk_size_gb" {
  description = "The size of disk used on dataproc worker node"
  type        = number
}

variable "image_version" {
  description = "The version of image used for dataproc"
  type        = string
}

variable "spark_shuffle_partitions" {
  description = "Default value for spark.sql.shuffle.partitions"
  type        = number
}

variable "spark_properties" {
  description = "Additional Dataproc software properties"
  type        = map(string)
}

variable "optional_components" {
  description = "Optional Dataproc packages to install"
  type        = list(string)
}

variable "initialization_actions" {
  type = list(object({
    script      = string
    timeout_sec = optional(number, 300)
  }))
}

variable "network_tags" {
  description = "Network tags added to Dataproc VMs"
  type        = list(string)
}

variable "subnetwork_uri" {
  description = "Subnetwork used by Dataproc cluster VMs"
  type        = string
}
