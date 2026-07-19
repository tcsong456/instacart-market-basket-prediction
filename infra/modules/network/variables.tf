variable "vpc_name" {
  type = string
}

variable "project_id" {
  type = string
}

variable "subnet_name" {
  type = string
}

variable "region" {
  type = string
}

variable "nat_router_name" {
  type = string
}

variable "nat_name" {
  type = string
}

variable "enable_iap_ssh" {
  description = "Whether to allow SSH to Dataproc VMs through IAP"
  type        = bool
}

variable "enable_nat_router" {
  description = "Whether to allow NAT router for outbound traffic"
  type        = bool
}

variable "dataproc_network_tag" {
  description = "Network tag assigned to Dataproc VMs and used by firewall rules"
  type        = string
}

variable "routing_mode" {
  description = "VPC dynamic routing mode"
  type        = string

  validation {
    condition = contains(
      ["REGIONAL", "GLOBAL"],
      var.routing_mode
    )
    error_message = "routing_mode must be either REGIONAL or GLOBAL."
  }
}

variable "subnet_cidr" {
  description = "Primary IPv4 CIDR range for the Dataproc subnet"
  type        = string

  validation {
    condition     = can(cidrhost(var.subnet_cidr, 0))
    error_message = "subnet_cidr must be a valid IPv4 CIDR block"
  }
}

variable "flow_log_aggregation_interval" {
  description = "Aggregation interval for VPC flow logs"
  type        = string

  validation {
    condition = contains(
      [
        "INTERVAL_5_SEC",
        "INTERVAL_30_SEC",
        "INTERVAL_1_MIN",
        "INTERVAL_5_MIN",
        "INTERVAL_10_MIN",
        "INTERVAL_15_MIN"
      ],
      var.flow_log_aggregation_interval
    )
    error_message = "Invalid VPC flow-log aggregation interval."
  }
}

variable "flow_log_sampling" {
  description = "Fraction of network flows sampled for VPC flow logs"
  type        = number

  validation {
    condition = (
      var.flow_log_sampling >= 0 &&
      var.flow_log_sampling <= 1
    )
    error_message = "flow_log_sampling must be between 0 and 1"
  }
}

variable "flow_log_metadata" {
  description = "Metadata mode for VPC flow logs"
  type        = string

  validation {
    condition = contains(
      [
        "EXCLUDE_ALL_METADATA",
        "INCLUDE_ALL_METADATA",
        "CUSTOM_METADATA",
      ],
      var.flow_log_metadata
    )
    error_message = "Invalid VPC flow-log metadata mode."
  }
}

variable "internal_firewall_priority" {
  description = "Priority of the Dataproc internal communication firewall rule"
  type        = number

  validation {
    condition = (
      var.internal_firewall_priority >= 0 &&
      var.internal_firewall_priority <= 65535
    )
    error_message = "internal_firewall_priority must be between 0 and 65535"
  }
}

variable "iap_firewall_priority" {
  description = "Priority of the IAP SSH firewall rule"
  type        = number

  validation {
    condition = (
      var.iap_firewall_priority >= 0 &&
      var.iap_firewall_priority <= 65535
    )
    error_message = "iap_firewall_priority must be between 0 and 65535."
  }
}

variable "nat_min_ports_per_vm" {
  description = "Minimum number of NAT source ports allocated per VM"
  type        = number

  validation {
    condition     = var.nat_min_ports_per_vm >= 32
    error_message = "nat_min_ports_per_vm must be at least 32."
  }
}