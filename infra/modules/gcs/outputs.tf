output "bucket_names" {
  description = "Names of created buckets"

  value = {
    for k, v in google_storage_bucket.bucket :
    k => v.name
  }
}

output "bucket_urls" {
  description = "URL of the created buckets"

  value = {
    for k, v in google_storage_bucket.bucket :
    k => v.url
  }
}