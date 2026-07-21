//! Resolve a user-facing transfer target into an [`ArtifactStore`].
//!
//! The client edge (CLI / Python binding) names each end of a transfer with one
//! string, a local directory path or an object-store URL, so the mapping from
//! that string to a store lives here, once, rather than in every frontend:
//!
//! - `s3://bucket/prefix` → [`ObjectArtifactStore`] over Amazon S3 or any
//!   S3-compatible endpoint (MinIO, R2, …). Credentials, region, and endpoint come
//!   from the standard `AWS_*` environment variables (`AWS_ACCESS_KEY_ID`,
//!   `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_ENDPOINT`, `AWS_ALLOW_HTTP`),
//!   read by `AmazonS3Builder::from_env`. The URL path becomes the per-tenant key
//!   prefix. Setting `AWS_COPY_IF_NOT_EXISTS` (e.g.
//!   `header: cf-copy-destination-if-none-match: *` on R2) additionally makes the
//!   large-artifact multipart claim atomic; without it, that one step falls back
//!   to a probe-then-copy on providers with no conditional copy.
//! - anything without a `://` scheme → [`LocalArtifactStore`] on that directory.
//!
//! Other object-store schemes (`gs://`, `az://`) are rejected until there is a
//! deployment that needs them. Adding one is a new match arm plus a Cargo
//! feature, never a change to the transfer code.

use crate::artifact_store::ArtifactStore;
use crate::error::{ArtifactStoreError, Result};
use crate::local_artifact_store::LocalArtifactStore;
use crate::object_artifact_store::ObjectArtifactStore;
use object_store::aws::AmazonS3Builder;
use std::sync::Arc;

/// Resolves `target` (a local directory path or an `s3://bucket/prefix` URL)
/// into an artifact store.
///
/// Local stores are opened without fsync, matching the engine's default
/// durability. Returns a `Backend` error for an unsupported URL scheme or an S3
/// URL with no bucket, and surfaces the S3 builder's own error when the `AWS_*`
/// environment is incomplete (e.g. a missing region).
pub fn artifact_store_from_target(target: &str) -> Result<Box<dyn ArtifactStore>> {
    if let Some(rest) = target.strip_prefix("s3://") {
        let (bucket, prefix) = rest.split_once('/').unwrap_or((rest, ""));
        if bucket.is_empty() {
            return Err(ArtifactStoreError::Backend(format!(
                "s3 target {target:?} has no bucket; expected s3://bucket[/prefix]"
            )));
        }
        let s3 = AmazonS3Builder::from_env()
            .with_bucket_name(bucket)
            .build()
            .map_err(|error| ArtifactStoreError::Backend(error.to_string()))?;
        let store = ObjectArtifactStore::new(Arc::new(s3), prefix.trim_matches('/'))?;
        return Ok(Box::new(store));
    }
    if target.contains("://") {
        return Err(ArtifactStoreError::Backend(format!(
            "unsupported target scheme in {target:?}; expected a local directory path or \
             s3://bucket/prefix"
        )));
    }
    Ok(Box::new(LocalArtifactStore::new(target, false)))
}
