output "bucket_urls" {
  description = "Urls of created buckets"
  value       = module.gcs_buckets.bucket_urls
}