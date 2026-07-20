resource "random_id" "bucket_suffix" {
  byte_length = 4
}

module "gcs_buckets" {
  project_id = var.project_id
  location   = var.region
  buckets    = var.buckets
  suffix     = random_id.bucket_suffix.hex
  prefix     = var.prefix

  source = "../../modules/gcs"

  depends_on = [google_project_service.required_apis]
}

module "iam" {
  source = "../../modules/iam"

  project_id         = var.project_id
  service_account_id = var.service_account_id
  display_name       = var.display_name
  admin_user         = var.admin_user
  enable_iap_ssh     = var.enable_iap_ssh

  project_roles = var.project_roles

  bucket_roles = {
    raw_admin = {
      bucket = module.gcs_buckets.bucket_names["raw"]
      role   = "roles/storage.objectAdmin"
    }

    bronze_admin = {
      bucket = module.gcs_buckets.bucket_names["bronze"]
      role   = "roles/storage.objectAdmin"
    }

    silver_admin = {
      bucket = module.gcs_buckets.bucket_names["silver"]
      role   = "roles/storage.objectAdmin"
    }

    gold_admin = {
      bucket = module.gcs_buckets.bucket_names["gold"]
      role   = "roles/storage.objectAdmin"
    }

    dataproc_staging_admin = {
      bucket = module.gcs_buckets.bucket_names["dataproc-staging"]
      role   = "roles/storage.objectAdmin"
    }
    dataproc_temp_admin = {
      bucket = module.gcs_buckets.bucket_names["dataproc-temp"]
      role   = "roles/storage.objectAdmin"
    }
  }

  depends_on = [google_project_service.required_apis]
}

module "dataproc" {
  source = "../../modules/dataproc"

  project_id   = var.project_id
  cluster_name = "dataproc-cluster"
  region       = var.region

  prefix                = var.prefix
  suffix                = random_id.bucket_suffix.hex
  network_tags          = [module.network.dataproc_network_tag]
  service_account_email = module.iam.service_account_email

  image_version  = var.image_version
  staging_bucket = module.gcs_buckets.bucket_names["dataproc-staging"]
  temp_bucket    = module.gcs_buckets.bucket_names["dataproc-temp"]

  spark_shuffle_partitions = var.spark_shuffle_partitions

  spark_properties    = var.spark_properties
  optional_components = var.optional_components

  worker_count             = var.worker_count
  master_machine_type      = var.master_machine_type
  boot_disk_type           = var.boot_disk_type
  master_boot_disk_size_gb = var.master_boot_disk_size_gb
  worker_machine_type      = var.worker_machine_type
  worker_boot_disk_size_gb = var.worker_boot_disk_size_gb

  initialization_actions = var.initialization_actions

  subnetwork_uri = module.network.subnet_self_link

  depends_on = [
    module.gcs_buckets,
    module.iam,
    google_project_service.required_apis,
    module.network
  ]
}

module "network" {
  source = "../../modules/network"

  project_id = var.project_id
  region     = var.region

  vpc_name     = "${var.prefix}-vpc"
  subnet_name  = "${var.prefix}-dataproc-subnet"
  routing_mode = "GLOBAL"
  subnet_cidr  = var.subnet_cidr

  enable_nat_router    = var.enable_nat_router
  nat_name             = "${var.prefix}-cloud-nat"
  nat_router_name      = "${var.prefix}-nat-router"
  nat_min_ports_per_vm = 64

  enable_iap_ssh = var.enable_iap_ssh

  dataproc_network_tag = "dataproc"

  flow_log_aggregation_interval = "INTERVAL_5_SEC"
  flow_log_sampling             = 0.5
  flow_log_metadata             = "INCLUDE_ALL_METADATA"

  internal_firewall_priority = 1000
  iap_firewall_priority      = 1000

  depends_on = [google_project_service.required_apis]
}