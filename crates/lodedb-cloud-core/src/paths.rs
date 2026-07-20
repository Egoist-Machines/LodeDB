//! Path-containment guard for filesystem-backed artifact stores.
//!
//! A committed generation's artifact *names* and pointer *keys* are joined onto a
//! store root to form on-disk paths. Once a name or key can originate from CLI or
//! remote/control-plane input (the cloud trust boundary), a crafted `..` segment
//! or absolute path could otherwise read or write outside the store root, a
//! cross-tenant disclosure in a multi-tenant store. [`resolve_within`] is the
//! single primitive that confines every such join to its root.

use crate::error::{ArtifactStoreError, Result};
use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

/// Canonicalises `candidate` and proves it stays within `root`.
///
/// Follows symlinks through the candidate's existing prefix (matching Python's
/// `Path.resolve`) and normalises `.`/`..` in any not-yet-created tail, then
/// asserts the result is `root` itself or beneath it. Resolving *first* is what
/// makes this sound: a purely lexical `starts_with` would read `root/../etc` as
/// "under root", and a symlink inside the tree could redirect outside it. An
/// absolute `candidate` discards `root` and resolves outside it, so it is
/// rejected too. Returns an `Integrity` error on any escape (defence against
/// CWE-22 path traversal).
///
/// `root` need not exist yet: a fresh backup destination resolves to the path it
/// will occupy (its deepest existing ancestor, canonicalised, plus the pending
/// tail), so a not-yet-created store directory is accepted and created on first
/// write rather than being rejected with `ENOENT`.
pub fn resolve_within(root: &Path, candidate: &Path) -> Result<PathBuf> {
    let root_real = canonical_resolve(root)?;
    let resolved = canonical_resolve(candidate)?;
    if resolved == root_real || resolved.starts_with(&root_real) {
        Ok(resolved)
    } else {
        Err(ArtifactStoreError::Integrity(format!(
            "path {candidate:?} escapes store root {root_real:?}"
        )))
    }
}

/// A path's canonical identity string: absolute, symlink-resolved (through the
/// existing prefix), `.`/`..`-normalised, the same resolution
/// [`resolve_within`] uses. Two spellings of one directory (relative vs
/// absolute, redundant `.` segments) map to one identity. On a resolution
/// failure the raw spelling is returned, which for the sidecar-trust caller can
/// only produce a false *mismatch* (fails toward force, never toward trusting
/// the wrong remote).
pub(crate) fn canonical_identity(candidate: &Path) -> String {
    // A bare relative path must anchor to the invocation cwd explicitly, or a
    // fully-nonexistent path would stay relative through the lexical fallback.
    let absolute = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        match std::env::current_dir() {
            Ok(cwd) => cwd.join(candidate),
            Err(_) => return candidate.to_string_lossy().into_owned(),
        }
    };
    match canonical_resolve(&absolute) {
        Ok(resolved) => resolved.to_string_lossy().into_owned(),
        Err(_) => candidate.to_string_lossy().into_owned(),
    }
}

/// Resolves `candidate` to an absolute, normalised path.
///
/// `std::fs::canonicalize` requires the whole path to exist, but artifact and
/// pointer paths are routinely resolved before they are created. So we
/// canonicalise the deepest existing ancestor (resolving symlinks like Python's
/// `resolve`) and re-append the not-yet-created tail, then collapse any `.`/`..`
/// lexically.
fn canonical_resolve(candidate: &Path) -> Result<PathBuf> {
    let mut existing = candidate.to_path_buf();
    let mut tail: Vec<OsString> = Vec::new();
    loop {
        if existing.exists() {
            let mut resolved = existing.canonicalize()?;
            for part in tail.iter().rev() {
                resolved.push(part);
            }
            return Ok(lexical_normalize(&resolved));
        }
        match existing.file_name() {
            Some(name) => {
                tail.push(name.to_os_string());
                if !existing.pop() {
                    break;
                }
            }
            None => break,
        }
    }
    Ok(lexical_normalize(candidate))
}

/// Collapses `.` and `..` components purely lexically (no filesystem access), so
/// a `..` in the not-yet-created tail cannot lift the path above the resolved
/// prefix without the containment check catching it.
fn lexical_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::resolve_within;
    use crate::error::ArtifactStoreError;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// A unique, created temp directory (std-only, no dev-dependency).
    fn temp_dir(label: &str) -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("orecloud-paths-{label}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn accepts_a_path_inside_the_root() {
        let root = temp_dir("inside");
        let resolved = resolve_within(&root, &root.join("idx.gen/g0.json")).unwrap();
        assert!(resolved.starts_with(root.canonicalize().unwrap()));
    }

    #[test]
    fn rejects_a_parent_traversal_escape() {
        let root = temp_dir("escape");
        let err = resolve_within(&root, &root.join("../secrets")).unwrap_err();
        assert!(matches!(err, ArtifactStoreError::Integrity(_)));
    }

    #[test]
    fn accepts_a_not_yet_created_root() {
        // A fresh backup destination that does not exist yet must resolve, not
        // fail with ENOENT; the store creates it on first write.
        let root = temp_dir("fresh-parent").join("new-backup");
        let resolved = resolve_within(&root, &root.join("idx.commit.json")).unwrap();
        assert!(resolved.ends_with("new-backup/idx.commit.json"));
    }
}
