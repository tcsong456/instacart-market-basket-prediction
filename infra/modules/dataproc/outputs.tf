output "cluster_id" {
  description = "Dataproc cluster resource ID"
  value       = google_dataproc_cluster.instacart_cluster.id
}

output "cluster_name" {
  description = "Dataproc cluster name"
  value       = google_dataproc_cluster.instacart_cluster.name
}