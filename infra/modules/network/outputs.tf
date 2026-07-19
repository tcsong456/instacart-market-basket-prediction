output "network_id" {
  description = "ID of the VPC network"
  value       = google_compute_network.instacart_vpc.id
}


output "network_name" {
  description = "Name of the VPC network"
  value       = google_compute_network.instacart_vpc.name
}


output "network_self_link" {
  description = "Self-link of the VPC network"
  value       = google_compute_network.instacart_vpc.self_link
}


output "subnet_id" {
  description = "ID of the Dataproc subnet"
  value       = google_compute_subnetwork.instacart_dataproc_subnet.id
}


output "subnet_name" {
  description = "Name of the Dataproc subnet"
  value       = google_compute_subnetwork.instacart_dataproc_subnet.name
}


output "subnet_self_link" {
  description = "Self-link of the Dataproc subnet"
  value       = google_compute_subnetwork.instacart_dataproc_subnet.self_link
}


output "subnet_cidr" {
  description = "CIDR range of the Dataproc subnet"
  value       = google_compute_subnetwork.instacart_dataproc_subnet.ip_cidr_range
}


output "dataproc_network_tag" {
  description = "Network tag that must be assigned to Dataproc VMs"
  value       = var.dataproc_network_tag
}


output "router_name" {
  description = "Name of the Cloud Router, or null when NAT router is disabled"
  value = (
    var.enable_nat_router
    ? google_compute_router.instacart_nat_router[0].name
    : null
  )
}


output "nat_name" {
  description = "Name of the Cloud NAT gateway, or null when NAT router is disabled"
  value = (
    var.enable_nat_router
    ? google_compute_router_nat.instacart_cloud_nat[0].name
    : null
  )
}