resource "google_compute_network" "instacart_vpc" {
  name                    = var.vpc_name
  project                 = var.project_id
  auto_create_subnetworks = false
  routing_mode            = var.routing_mode

  description = "VPC network for private Dataproc workloads"
}

resource "google_compute_subnetwork" "instacart_dataproc_subnet" {
  description = "Private subnet for Dataproc cluster nodes"
  project     = var.project_id

  name          = var.subnet_name
  region        = var.region
  network       = google_compute_network.instacart_vpc.id
  ip_cidr_range = var.subnet_cidr

  private_ip_google_access = true

  log_config {
    aggregation_interval = var.flow_log_aggregation_interval
    flow_sampling        = var.flow_log_sampling
    metadata             = var.flow_log_metadata
  }
}

resource "google_compute_firewall" "instacart_internal_traffic" {
  description = "Allow internal communication between Dataproc cluster nodes"
  project     = var.project_id

  name      = "${var.vpc_name}-internal-allowed-traffic"
  network   = google_compute_network.instacart_vpc.name
  direction = "INGRESS"
  priority  = var.internal_firewall_priority

  source_tags = [var.dataproc_network_tag]
  target_tags = [var.dataproc_network_tag]

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "icmp"
  }
}

resource "google_compute_firewall" "instacart_iap_ssh" {
  count       = var.enable_iap_ssh ? 1 : 0
  project     = var.project_id
  description = "Allow SSH through Google Cloud IAP"

  name      = "${var.vpc_name}-allowed-iap-ssh"
  network   = google_compute_network.instacart_vpc.name
  direction = "INGRESS"
  priority  = var.iap_firewall_priority

  source_ranges = ["35.235.240.0/20"]
  target_tags   = [var.dataproc_network_tag]

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

resource "google_compute_router" "instacart_nat_router" {
  count   = var.enable_nat_router ? 1 : 0
  project = var.project_id

  name    = var.nat_router_name
  region  = var.region
  network = google_compute_network.instacart_vpc.id

  description = "Cloud Router for Dataproc Cloud NAT"
}

resource "google_compute_router_nat" "instacart_cloud_nat" {
  count   = var.enable_nat_router ? 1 : 0
  project = var.project_id

  name   = var.nat_name
  region = var.region
  router = google_compute_router.instacart_nat_router[0].name

  nat_ip_allocate_option = "AUTO_ONLY"

  source_subnetwork_ip_ranges_to_nat = "LIST_OF_SUBNETWORKS"

  subnetwork {
    name                    = google_compute_subnetwork.instacart_dataproc_subnet.id
    source_ip_ranges_to_nat = ["ALL_IP_RANGES"]
  }

  min_ports_per_vm = var.nat_min_ports_per_vm

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}
