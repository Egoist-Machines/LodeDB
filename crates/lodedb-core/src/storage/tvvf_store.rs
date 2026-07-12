//! Original-precision vector sidecar with positional row reads.
//!
//! Unlike the other sidecars, this module intentionally retains only ID indexes
//! and one open file handle per segment in RAM, then reads candidate payloads by
//! absolute file offset. The retained handles keep a reader's committed snapshot
//! readable when a later writer collects its unlinked epoch.

use crate::error::CoreError;
use crate::storage::commit_manifest::generation_dir;
use crate::storage::util::{
    f16_bits_to_f32, f32_to_f16_bits, fnv1a64, fnv1a64_update, py_canonical_json,
    write_pretty_json_atomic,
};
use rayon::prelude::*;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub const TVVF_SCHEMA_VERSION: u64 = 1;
pub const TVVF_BASE_MAGIC: &[u8; 8] = b"EEVFB001";
pub const TVVF_DELTA_MAGIC: &[u8; 8] = b"EEVFD001";
pub const TVVF_DELTA_DIR_SUFFIX: &str = ".tvvf-delta";
pub const TVVF_DELTA_MANIFEST_NAME: &str = "manifest.json";

// Limit a fetch to a modest number of parallel positional-read streams. A large
// base segment commonly supplies every candidate, so segment-level parallelism
// alone would otherwise leave its scattered reads serial.
const TVVF_MAX_PARALLEL_RUN_TASKS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TvvfDtype {
    Float16,
    Float32,
    Int8,
}

impl TvvfDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Float16 => "float16",
            Self::Float32 => "float32",
            Self::Int8 => "int8",
        }
    }
}

impl AsRef<str> for TvvfDtype {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

#[derive(Debug)]
pub enum TvvfError {
    Io(io::Error),
    Corrupt(String),
    Invalid(String),
}

impl TvvfError {
    pub fn is_corrupt(&self) -> bool {
        matches!(self, Self::Corrupt(_))
    }
}

impl Display for TvvfError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "tvvf I/O error: {error}"),
            Self::Corrupt(message) => write!(f, "corrupt tvvf sidecar: {message}"),
            Self::Invalid(message) => write!(f, "invalid tvvf input: {message}"),
        }
    }
}

impl std::error::Error for TvvfError {}

impl From<io::Error> for TvvfError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

pub type TvvfResult<T> = Result<T, TvvfError>;
pub type TvvfManifestEntry = Value;

pub fn base_path(dir: impl AsRef<Path>, index_key: &str, vf_epoch: u64) -> PathBuf {
    generation_dir(dir.as_ref(), index_key).join(format!("vf{vf_epoch}.tvvf"))
}

pub fn manifest_path(dir: impl AsRef<Path>, index_key: &str, vf_epoch: u64) -> PathBuf {
    let base = base_path(dir, index_key, vf_epoch);
    delta_dir(&base).join(TVVF_DELTA_MANIFEST_NAME)
}

pub fn record_base<D: AsRef<str>>(
    dir: impl AsRef<Path>,
    index_key: &str,
    vf_epoch: u64,
    dtype: D,
    dim: usize,
    rows: &[(u64, &[f32])],
) -> TvvfResult<TvvfManifestEntry> {
    record_base_with_fsync(dir, index_key, vf_epoch, dtype, dim, rows, false)
}

pub fn record_base_with_fsync<D: AsRef<str>>(
    dir: impl AsRef<Path>,
    index_key: &str,
    vf_epoch: u64,
    dtype: D,
    dim: usize,
    rows: &[(u64, &[f32])],
    fsync: bool,
) -> TvvfResult<TvvfManifestEntry> {
    if index_key.is_empty() {
        return Err(invalid("tvvf index key must not be empty"));
    }
    let dtype = parse_dtype(dtype.as_ref())?;
    validate_rows(dtype, dim, rows)?;
    let base = base_path(dir, index_key, vf_epoch);
    let mut ids = |emit: &mut dyn FnMut(u64) -> TvvfResult<()>| {
        for (id, _) in rows {
            emit(*id)?;
        }
        Ok(())
    };
    let mut payloads = |emit: &mut dyn FnMut(&[u8]) -> TvvfResult<()>| {
        for (_, row) in rows {
            emit(&encode_row(dtype, dim, row)?)?;
        }
        Ok(())
    };
    let written = write_segment(
        &base,
        TVVF_BASE_MAGIC,
        dtype,
        dim,
        rows.len(),
        &[],
        fsync,
        &mut ids,
        &mut payloads,
    )?;
    write_base_manifest(
        &base,
        index_key,
        vf_epoch,
        dtype,
        dim,
        rows.len(),
        written,
        fsync,
    )
}

/// Records a base from rows already encoded with [`encode_row`]. This stays
/// crate-visible because the engine captures caller vectors before its lossy
/// TurboVec path and holds the encoded payloads until the commit is sealed.
pub(crate) fn record_encoded_base_with_fsync(
    dir: impl AsRef<Path>,
    index_key: &str,
    vf_epoch: u64,
    dtype: TvvfDtype,
    dim: usize,
    rows: &[(u64, &[u8])],
    fsync: bool,
) -> TvvfResult<TvvfManifestEntry> {
    if index_key.is_empty() {
        return Err(invalid("tvvf index key must not be empty"));
    }
    validate_encoded_rows(dtype, dim, rows)?;
    let base = base_path(dir, index_key, vf_epoch);
    let mut ids = |emit: &mut dyn FnMut(u64) -> TvvfResult<()>| {
        for (id, _) in rows {
            emit(*id)?;
        }
        Ok(())
    };
    let mut payloads = |emit: &mut dyn FnMut(&[u8]) -> TvvfResult<()>| {
        for (_, row) in rows {
            emit(row)?;
        }
        Ok(())
    };
    let written = write_segment(
        &base,
        TVVF_BASE_MAGIC,
        dtype,
        dim,
        rows.len(),
        &[],
        fsync,
        &mut ids,
        &mut payloads,
    )?;
    write_base_manifest(&base, index_key, vf_epoch, dtype, dim, rows.len(), written, fsync)
}

pub fn append_delta(
    dir: impl AsRef<Path>,
    index_key: &str,
    vf_epoch: u64,
    upserts: &[(u64, &[f32])],
    deleted: &[u64],
) -> TvvfResult<TvvfManifestEntry> {
    append_delta_with_fsync(dir, index_key, vf_epoch, upserts, deleted, false)
}

pub fn append_delta_with_fsync(
    dir: impl AsRef<Path>,
    index_key: &str,
    vf_epoch: u64,
    upserts: &[(u64, &[f32])],
    deleted: &[u64],
    fsync: bool,
) -> TvvfResult<TvvfManifestEntry> {
    let base = base_path(&dir, index_key, vf_epoch);
    let base_header = read_header(&base, TVVF_BASE_MAGIC, false)?;
    validate_rows(base_header.dtype, base_header.dim, upserts)?;
    let deleted = validate_deleted(deleted, upserts.iter().map(|(id, _)| *id))?;
    let manifest_file = manifest_path(dir, index_key, vf_epoch);
    let mut manifest = read_manifest(&manifest_file)?;
    validate_manifest_identity(&manifest, index_key, vf_epoch)?;
    let sequence = json_object(&manifest, "tvvf manifest")?
        .get("next_seq")
        .and_then(Value::as_u64)
        .ok_or_else(|| corrupt("tvvf manifest is missing next_seq"))?;
    let next_sequence = sequence
        .checked_add(1)
        .ok_or_else(|| corrupt("tvvf manifest next_seq overflow"))?;
    let file_name = format!("delta-{sequence:08}.tvfd");
    let segment = delta_dir(&base).join(&file_name);
    let mut ids = |emit: &mut dyn FnMut(u64) -> TvvfResult<()>| {
        for (id, _) in upserts {
            emit(*id)?;
        }
        Ok(())
    };
    let mut payloads = |emit: &mut dyn FnMut(&[u8]) -> TvvfResult<()>| {
        for (_, row) in upserts {
            emit(&encode_row(base_header.dtype, base_header.dim, row)?)?;
        }
        Ok(())
    };
    let written = write_segment(
        &segment,
        TVVF_DELTA_MAGIC,
        base_header.dtype,
        base_header.dim,
        upserts.len(),
        &deleted,
        fsync,
        &mut ids,
        &mut payloads,
    )?;
    let object = manifest
        .as_object_mut()
        .ok_or_else(|| corrupt("tvvf manifest must be an object"))?;
    object
        .get_mut("deltas")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| corrupt("tvvf manifest deltas must be an array"))?
        .push(serde_json::json!({
            "file_name": file_name,
            "sha256": written.sha256,
            "file_bytes": written.file_bytes,
            "seq": sequence,
            "upsert_rows": upserts.len(),
            "deleted_rows": deleted.len(),
        }));
    object.insert("next_seq".to_string(), Value::from(next_sequence));
    write_manifest(&manifest_file, &manifest, fsync)?;
    Ok(manifest)
}

/// Appends a delta from rows already encoded with [`encode_row`]. See
/// [`record_encoded_base_with_fsync`] for why this is crate-visible.
pub(crate) fn append_encoded_delta_with_fsync(
    dir: impl AsRef<Path>,
    index_key: &str,
    vf_epoch: u64,
    upserts: &[(u64, &[u8])],
    deleted: &[u64],
    fsync: bool,
) -> TvvfResult<TvvfManifestEntry> {
    let base = base_path(&dir, index_key, vf_epoch);
    let base_header = read_header(&base, TVVF_BASE_MAGIC, false)?;
    validate_encoded_rows(base_header.dtype, base_header.dim, upserts)?;
    let deleted = validate_deleted(deleted, upserts.iter().map(|(id, _)| *id))?;
    let manifest_file = manifest_path(dir, index_key, vf_epoch);
    let mut manifest = read_manifest(&manifest_file)?;
    validate_manifest_identity(&manifest, index_key, vf_epoch)?;
    let sequence = json_object(&manifest, "tvvf manifest")?
        .get("next_seq")
        .and_then(Value::as_u64)
        .ok_or_else(|| corrupt("tvvf manifest is missing next_seq"))?;
    let next_sequence = sequence
        .checked_add(1)
        .ok_or_else(|| corrupt("tvvf manifest next_seq overflow"))?;
    let file_name = format!("delta-{sequence:08}.tvfd");
    let segment = delta_dir(&base).join(&file_name);
    let mut ids = |emit: &mut dyn FnMut(u64) -> TvvfResult<()>| {
        for (id, _) in upserts {
            emit(*id)?;
        }
        Ok(())
    };
    let mut payloads = |emit: &mut dyn FnMut(&[u8]) -> TvvfResult<()>| {
        for (_, row) in upserts {
            emit(row)?;
        }
        Ok(())
    };
    let written = write_segment(
        &segment,
        TVVF_DELTA_MAGIC,
        base_header.dtype,
        base_header.dim,
        upserts.len(),
        &deleted,
        fsync,
        &mut ids,
        &mut payloads,
    )?;
    let object = manifest
        .as_object_mut()
        .ok_or_else(|| corrupt("tvvf manifest must be an object"))?;
    object
        .get_mut("deltas")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| corrupt("tvvf manifest deltas must be an array"))?
        .push(serde_json::json!({
            "file_name": file_name,
            "sha256": written.sha256,
            "file_bytes": written.file_bytes,
            "seq": sequence,
            "upsert_rows": upserts.len(),
            "deleted_rows": deleted.len(),
        }));
    object.insert("next_seq".to_string(), Value::from(next_sequence));
    write_manifest(&manifest_file, &manifest, fsync)?;
    Ok(manifest)
}

/// Restores the sidecar-local manifest from the generation root's committed tvvf
/// entry before appending. The root manifest is the commit point, so a crash after
/// a sidecar segment/local-manifest write but before the root swap must not let a
/// later append accidentally promote that orphaned segment.
pub(crate) fn restore_manifest_with_fsync(
    dir: impl AsRef<Path>,
    index_key: &str,
    vf_epoch: u64,
    manifest: &TvvfManifestEntry,
    fsync: bool,
) -> TvvfResult<()> {
    validate_manifest_identity(manifest, index_key, vf_epoch)?;
    write_manifest(&manifest_path(dir, index_key, vf_epoch), manifest, fsync)
}

/// Loads only the newest delta's ID and tombstone index. This deliberately does
/// not open the base or older delta segments, so a resident reader can advance
/// on the append path in O(the new segment).
pub(crate) fn load_latest_delta_segment(
    dir: impl AsRef<Path>,
    index_key: &str,
    vf_epoch: u64,
    manifest: &TvvfManifestEntry,
) -> TvvfResult<TvvfDeltaSegment> {
    validate_manifest_identity(manifest, index_key, vf_epoch)?;
    let entry = json_object(manifest, "tvvf manifest")?
        .get("deltas")
        .and_then(Value::as_array)
        .and_then(|deltas| deltas.last())
        .ok_or_else(|| corrupt("tvvf manifest has no delta segment to load"))?;
    let entry = json_object(entry, "tvvf delta manifest entry")?;
    let sequence = entry
        .get("seq")
        .and_then(Value::as_u64)
        .ok_or_else(|| corrupt("tvvf delta entry is missing seq"))?;
    let name = entry
        .get("file_name")
        .and_then(Value::as_str)
        .filter(|name| Path::new(name).file_name() == Some(std::ffi::OsStr::new(name)))
        .ok_or_else(|| corrupt("tvvf delta entry has invalid file_name"))?;
    let expected = entry
        .get("sha256")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| corrupt("tvvf delta entry is missing sha256"))?;
    let base = base_path(dir, index_key, vf_epoch);
    let path = delta_dir(&base).join(name);
    Ok(TvvfDeltaSegment {
        sequence,
        name: name.to_string(),
        // Open before validating so the attached segment and every byte checked
        // below are from one inode even if a later epoch GC unlinks the pathname.
        index: read_index(&path, TVVF_DELTA_MAGIC, true, Some(expected))?,
    })
}

pub fn fold(
    dir: impl AsRef<Path>,
    index_key: &str,
    vf_epoch: u64,
) -> TvvfResult<TvvfManifestEntry> {
    fold_with_fsync(dir, index_key, vf_epoch, false)
}

pub fn fold_with_fsync(
    dir: impl AsRef<Path>,
    index_key: &str,
    vf_epoch: u64,
    fsync: bool,
) -> TvvfResult<TvvfManifestEntry> {
    let root = dir.as_ref();
    let manifest = read_manifest(&manifest_path(root, index_key, vf_epoch))?;
    validate_manifest_identity(&manifest, index_key, vf_epoch)?;
    let reader = TvvfReader::open(root, &manifest)?;
    let latest = latest_rows(&reader);
    let count = fold_count(&reader, &latest)?;
    let epoch = vf_epoch
        .checked_add(1)
        .ok_or_else(|| invalid("tvvf epoch overflow"))?;
    let destination = base_path(root, index_key, epoch);
    let mut ids =
        |emit: &mut dyn FnMut(u64) -> TvvfResult<()>| visit_fold_ids(&reader, &latest, emit);
    let mut payloads =
        |emit: &mut dyn FnMut(&[u8]) -> TvvfResult<()>| visit_fold_rows(&reader, &latest, emit);
    let written = write_segment(
        &destination,
        TVVF_BASE_MAGIC,
        reader.dtype,
        reader.dim,
        count,
        &[],
        fsync,
        &mut ids,
        &mut payloads,
    )?;
    write_base_manifest(
        &destination,
        index_key,
        epoch,
        reader.dtype,
        reader.dim,
        count,
        written,
        fsync,
    )
}

#[derive(Debug)]
pub struct TvvfReader {
    index_key: String,
    vf_epoch: u64,
    dtype: TvvfDtype,
    dim: usize,
    base: SegmentIndex,
    deltas: Vec<SegmentIndex>,
    last_delta_sequence: Option<u64>,
    /// Latest live/tombstoned state for ids touched by delta segments. Keeping
    /// this alongside the segment indexes lets an appended segment update
    /// coverage without revisiting the base or older deltas.
    delta_states: HashMap<u64, bool>,
    live_rows: u64,
    tombstones: u64,
    corrupt_rows_seen: AtomicU64,
}

impl Clone for TvvfReader {
    fn clone(&self) -> Self {
        Self {
            index_key: self.index_key.clone(),
            vf_epoch: self.vf_epoch,
            dtype: self.dtype,
            dim: self.dim,
            base: self.base.clone(),
            deltas: self.deltas.clone(),
            last_delta_sequence: self.last_delta_sequence,
            delta_states: self.delta_states.clone(),
            live_rows: self.live_rows,
            tombstones: self.tombstones,
            corrupt_rows_seen: AtomicU64::new(self.corrupt_rows_seen()),
        }
    }
}

impl TvvfReader {
    pub fn open(dir: impl AsRef<Path>, manifest: &TvvfManifestEntry) -> TvvfResult<Self> {
        let root = dir.as_ref();
        let (index_key, vf_epoch) = validate_manifest_identity(manifest, "", 0)?;
        let base_file = base_path(root, &index_key, vf_epoch);
        let base = read_index(&base_file, TVVF_BASE_MAGIC, false, None)?;
        validate_manifest_base(manifest, &base_file, &base)?;
        let mut previous = None;
        let mut deltas = Vec::new();
        for entry in json_object(manifest, "tvvf manifest")?
            .get("deltas")
            .and_then(Value::as_array)
            .ok_or_else(|| corrupt("tvvf manifest deltas must be an array"))?
        {
            let entry = json_object(entry, "tvvf delta manifest entry")?;
            let sequence = entry
                .get("seq")
                .and_then(Value::as_u64)
                .ok_or_else(|| corrupt("tvvf delta entry is missing seq"))?;
            if previous.is_some_and(|value| sequence <= value) {
                return Err(corrupt("tvvf manifest has out-of-order segments"));
            }
            previous = Some(sequence);
            let name = entry
                .get("file_name")
                .and_then(Value::as_str)
                .filter(|name| Path::new(name).file_name() == Some(std::ffi::OsStr::new(name)))
                .ok_or_else(|| corrupt("tvvf delta entry has invalid file_name"))?;
            let expected = entry
                .get("sha256")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| corrupt("tvvf delta entry is missing sha256"))?;
            let path = delta_dir(&base_file).join(name);
            // read_index opens first, then validates its checksum, header, and
            // identifiers through the retained descriptor. This makes an open
            // reader immune to a concurrent GC unlink after the open succeeds.
            let segment = read_index(&path, TVVF_DELTA_MAGIC, true, Some(expected))?;
            validate_delta_segment(&segment, &base, name)?;
            deltas.push(segment);
        }
        let (live_rows, tombstones, delta_states) = coverage(&base, &deltas);
        Ok(Self {
            index_key,
            vf_epoch,
            dtype: base.dtype,
            dim: base.dim,
            base,
            deltas,
            last_delta_sequence: previous,
            delta_states,
            live_rows,
            tombstones,
            corrupt_rows_seen: AtomicU64::new(0),
        })
    }

    pub fn dtype(&self) -> TvvfDtype {
        self.dtype
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn vf_epoch(&self) -> u64 {
        self.vf_epoch
    }

    pub fn index_key(&self) -> &str {
        &self.index_key
    }

    pub fn coverage(&self) -> (u64, u64) {
        (self.live_rows, self.tombstones)
    }

    pub fn corrupt_rows_seen(&self) -> u64 {
        self.corrupt_rows_seen.load(Ordering::Relaxed)
    }

    /// Installs the just-appended delta segment without reopening and indexing
    /// the base or older delta files. The caller supplies the newest manifest
    /// entry, so sequence order remains part of the resident-reader invariant.
    pub(crate) fn append_delta_segment(
        &mut self,
        segment: TvvfDeltaSegment,
    ) -> TvvfResult<()> {
        if self
            .last_delta_sequence
            .is_some_and(|previous| segment.sequence <= previous)
        {
            return Err(corrupt("tvvf resident reader received an out-of-order segment"));
        }
        validate_delta_segment(&segment.index, &self.base, &segment.name)?;
        self.apply_delta_coverage(&segment.index);
        self.last_delta_sequence = Some(segment.sequence);
        self.deltas.push(segment.index);
        Ok(())
    }

    pub fn fetch_rows(&self, ids: &[u64]) -> Vec<Option<Vec<f32>>> {
        let mut groups = BTreeMap::<usize, BTreeMap<usize, Vec<usize>>>::new();
        for (position, id) in ids.iter().copied().enumerate() {
            if let Some((segment, row)) = self.resolve(id) {
                groups
                    .entry(segment)
                    .or_default()
                    .entry(row)
                    .or_default()
                    .push(position);
            }
        }
        let batches = groups
            .into_iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(slot, rows)| {
                let segment = if slot == 0 {
                    &self.base
                } else {
                    &self.deltas[slot - 1]
                };
                self.fetch_segment(segment, rows)
            })
            .collect::<Vec<_>>();
        let mut output = vec![None; ids.len()];
        for batch in batches {
            for (position, row) in batch {
                output[position] = row;
            }
        }
        output
    }

    fn resolve(&self, id: u64) -> Option<(usize, usize)> {
        for (index, segment) in self.deltas.iter().enumerate().rev() {
            if segment.deleted(id) {
                return None;
            }
            if let Some(row) = segment.row_for(id) {
                return Some((index + 1, row));
            }
        }
        self.base.row_for(id).map(|row| (0, row))
    }

    fn apply_delta_coverage(&mut self, segment: &SegmentIndex) {
        for id in &segment.deleted_ids {
            let prior = self.delta_states.get(id).copied();
            let was_live = prior.unwrap_or_else(|| self.base.row_for(*id).is_some());
            if was_live {
                self.live_rows = self.live_rows.saturating_sub(1);
            }
            if prior != Some(false) {
                self.tombstones = self.tombstones.saturating_add(1);
            }
            self.delta_states.insert(*id, false);
        }
        for id in &segment.ids {
            let prior = self.delta_states.get(id).copied();
            let was_live = prior.unwrap_or_else(|| self.base.row_for(*id).is_some());
            if !was_live {
                self.live_rows = self.live_rows.saturating_add(1);
            }
            if prior == Some(false) {
                self.tombstones = self.tombstones.saturating_sub(1);
            }
            self.delta_states.insert(*id, true);
        }
    }

    fn fetch_segment(
        &self,
        segment: &SegmentIndex,
        rows: BTreeMap<usize, Vec<usize>>,
    ) -> Vec<(usize, Option<Vec<f32>>)> {
        let rows = rows.into_iter().collect::<Vec<_>>();
        let mut runs = Vec::new();
        let mut start = 0;
        while start < rows.len() {
            let mut end = start + 1;
            while end < rows.len() && rows[end].0 <= rows[end - 1].0 + 1 {
                end += 1;
            }
            runs.push(start..end);
            start = end;
        }

        let run_chunk_len =
            (runs.len() + TVVF_MAX_PARALLEL_RUN_TASKS - 1) / TVVF_MAX_PARALLEL_RUN_TASKS;
        runs.par_chunks(run_chunk_len)
            .map(|chunk| self.fetch_run_chunk(segment, &rows, chunk))
            .reduce(Vec::new, |mut output, batch| {
                output.extend(batch);
                output
            })
    }

    fn fetch_run_chunk(
        &self,
        segment: &SegmentIndex,
        rows: &[(usize, Vec<usize>)],
        runs: &[std::ops::Range<usize>],
    ) -> Vec<(usize, Option<Vec<f32>>)> {
        // Each Rayon task clones the reader-held handle because Windows positional
        // reads move the file cursor. The held handle preserves this snapshot's
        // segment after a later epoch GC unlinks its path on Unix.
        let Ok(file) = segment.file.try_clone() else {
            self.corrupt_rows_seen.fetch_add(
                runs.iter().map(|run| run.len() as u64).sum::<u64>(),
                Ordering::Relaxed,
            );
            return runs
                .into_iter()
                .flat_map(|run| rows[run.clone()].iter())
                .flat_map(|(_, positions)| {
                    positions.iter().copied().map(|position| (position, None))
                })
                .collect();
        };
        let mut output = Vec::new();
        for run in runs {
            let first = rows[run.start].0;
            let count = run.len();
            let mut checksums = vec![0_u8; count * 4];
            let mut payloads = vec![0_u8; count * segment.row_stride];
            let ok = read_exact_at(
                &file,
                &mut checksums,
                segment.checksums_offset + first as u64 * 4,
            )
            .is_ok()
                && read_exact_at(
                    &file,
                    &mut payloads,
                    segment.rows_offset + first as u64 * segment.row_stride as u64,
                )
                .is_ok();
            if !ok {
                self.corrupt_rows_seen
                    .fetch_add(count as u64, Ordering::Relaxed);
                for (_, positions) in &rows[run.clone()] {
                    output.extend(positions.iter().copied().map(|position| (position, None)));
                }
                continue;
            }
            for (offset, (_, positions)) in rows[run.clone()].iter().enumerate() {
                let payload =
                    &payloads[offset * segment.row_stride..(offset + 1) * segment.row_stride];
                let checksum = u32::from_le_bytes(
                    checksums[offset * 4..offset * 4 + 4]
                        .try_into()
                        .expect("u32 checksum"),
                );
                let value = if row_checksum(payload) == checksum {
                    Some(decode_row(segment.dtype, segment.dim, payload))
                } else {
                    self.corrupt_rows_seen.fetch_add(1, Ordering::Relaxed);
                    None
                };
                output.extend(
                    positions
                        .iter()
                        .copied()
                        .map(|position| (position, value.clone())),
                );
            }
        }
        output
    }
}

#[derive(Debug, Clone)]
struct SegmentIndex {
    path: PathBuf,
    /// Kept for the lifetime of the reader so epoch GC cannot invalidate an open
    /// snapshot's positional reads after unlinking this segment on Unix.
    file: Arc<File>,
    dtype: TvvfDtype,
    dim: usize,
    row_stride: usize,
    ids: Vec<u64>,
    lookup_rows: Vec<u32>,
    deleted_ids: Vec<u64>,
    checksums_offset: u64,
    rows_offset: u64,
}

/// An indexed delta segment ready to attach to a resident [`TvvfReader`].
/// Kept crate-visible so the engine can move it directly from the append path
/// into the reader without exposing the raw segment representation publicly.
#[derive(Debug)]
pub(crate) struct TvvfDeltaSegment {
    sequence: u64,
    name: String,
    index: SegmentIndex,
}

impl SegmentIndex {
    fn row_for(&self, id: u64) -> Option<usize> {
        self.lookup_rows
            .binary_search_by_key(&id, |row| self.ids[*row as usize])
            .ok()
            .map(|position| self.lookup_rows[position] as usize)
    }

    fn deleted(&self, id: u64) -> bool {
        self.deleted_ids.binary_search(&id).is_ok()
    }
}

struct Header {
    dtype: TvvfDtype,
    dim: usize,
    row_count: usize,
    ids_checksum: u64,
    deleted_ids: Vec<u64>,
    ids_offset: u64,
    checksums_offset: u64,
    rows_offset: u64,
    row_stride: usize,
}

struct Written {
    sha256: String,
    file_bytes: u64,
    ids_checksum: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Latest {
    Deleted,
    Row(usize, usize),
}

fn parse_dtype(dtype: &str) -> TvvfResult<TvvfDtype> {
    match dtype {
        "float16" => Ok(TvvfDtype::Float16),
        "float32" => Ok(TvvfDtype::Float32),
        "int8" => Ok(TvvfDtype::Int8),
        _ => Err(invalid(format!("unsupported tvvf dtype {dtype:?}"))),
    }
}

fn validate_rows(dtype: TvvfDtype, dim: usize, rows: &[(u64, &[f32])]) -> TvvfResult<()> {
    if dim == 0 {
        return Err(invalid("tvvf dim must be positive"));
    }
    if rows.len() > u32::MAX as usize {
        return Err(invalid("tvvf row count exceeds the u32 lookup index limit"));
    }
    let mut ids = HashSet::with_capacity(rows.len());
    for (id, row) in rows {
        if row.len() != dim {
            return Err(invalid(format!("tvvf row {id} has unexpected dim")));
        }
        if !ids.insert(*id) {
            return Err(invalid(format!(
                "tvvf input contains duplicate stable id {id}"
            )));
        }
        if dtype == TvvfDtype::Int8 && row.iter().any(|value| !value.is_finite()) {
            return Err(invalid("int8 tvvf rows must contain only finite values"));
        }
    }
    Ok(())
}

fn validate_encoded_rows(dtype: TvvfDtype, dim: usize, rows: &[(u64, &[u8])]) -> TvvfResult<()> {
    if dim == 0 {
        return Err(invalid("tvvf dim must be positive"));
    }
    if rows.len() > u32::MAX as usize {
        return Err(invalid("tvvf row count exceeds the u32 lookup index limit"));
    }
    let stride = row_stride(dtype, dim)?;
    let mut ids = HashSet::with_capacity(rows.len());
    for (id, row) in rows {
        if row.len() != stride {
            return Err(invalid(format!("tvvf encoded row {id} has unexpected width")));
        }
        if !ids.insert(*id) {
            return Err(invalid(format!(
                "tvvf input contains duplicate stable id {id}"
            )));
        }
    }
    Ok(())
}

fn validate_deleted(
    deleted: &[u64],
    upserts: impl IntoIterator<Item = u64>,
) -> TvvfResult<Vec<u64>> {
    let upserts = upserts.into_iter().collect::<HashSet<_>>();
    let mut deleted = deleted.to_vec();
    deleted.sort_unstable();
    if deleted.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(invalid("tvvf delta contains duplicate tombstones"));
    }
    if deleted.iter().any(|id| upserts.contains(id)) {
        return Err(invalid("tvvf delta both upserts and deletes a stable id"));
    }
    Ok(deleted)
}

fn row_stride(dtype: TvvfDtype, dim: usize) -> TvvfResult<usize> {
    match dtype {
        TvvfDtype::Float16 => dim.checked_mul(2),
        TvvfDtype::Float32 => dim.checked_mul(4),
        TvvfDtype::Int8 => dim.checked_add(4),
    }
    .ok_or_else(|| invalid("tvvf row stride overflow"))
}

pub(crate) fn encode_row(dtype: TvvfDtype, dim: usize, row: &[f32]) -> TvvfResult<Vec<u8>> {
    let mut output = Vec::with_capacity(row_stride(dtype, dim)?);
    match dtype {
        TvvfDtype::Float16 => {
            for value in row {
                output.extend_from_slice(&f32_to_f16_bits(*value).to_le_bytes());
            }
        }
        TvvfDtype::Float32 => {
            for value in row {
                output.extend_from_slice(&value.to_le_bytes());
            }
        }
        TvvfDtype::Int8 => {
            let scale = row.iter().fold(0.0_f32, |max, value| max.max(value.abs()));
            let safe = if scale == 0.0 { 1.0 } else { scale };
            output.extend_from_slice(&scale.to_le_bytes());
            for value in row {
                let code = round_ties_even(*value / safe * 127.0).clamp(-127.0, 127.0);
                output.push(code as i8 as u8);
            }
        }
    }
    Ok(output)
}

/// Decodes an engine-held row captured by [`encode_row`]. Persisted rows are
/// checksum-verified by [`TvvfReader::fetch_rows`] before reaching this helper.
pub(crate) fn decode_row(dtype: TvvfDtype, dim: usize, bytes: &[u8]) -> Vec<f32> {
    match dtype {
        TvvfDtype::Float16 => bytes
            .chunks_exact(2)
            .map(|value| f16_bits_to_f32(u16::from_le_bytes([value[0], value[1]])))
            .collect(),
        TvvfDtype::Float32 => bytes
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes([value[0], value[1], value[2], value[3]]))
            .collect(),
        TvvfDtype::Int8 => {
            let scale = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            bytes[4..4 + dim]
                .iter()
                .map(|code| *code as i8 as f32 * scale / 127.0)
                .collect()
        }
    }
}

fn round_ties_even(value: f32) -> f32 {
    let floor = value.floor();
    let fraction = value - floor;
    if fraction > 0.5 || (fraction == 0.5 && (floor as i64).unsigned_abs() % 2 == 1) {
        floor + 1.0
    } else {
        floor
    }
}

fn row_checksum(bytes: &[u8]) -> u32 {
    fnv1a64(bytes) as u32
}

fn delta_dir(base: &Path) -> PathBuf {
    base.with_file_name(format!(
        "{}{}",
        base.file_name().unwrap_or_default().to_string_lossy(),
        TVVF_DELTA_DIR_SUFFIX
    ))
}

fn write_segment<I, R>(
    path: &Path,
    magic: &[u8; 8],
    dtype: TvvfDtype,
    dim: usize,
    rows: usize,
    deleted: &[u64],
    fsync: bool,
    ids: &mut I,
    payloads: &mut R,
) -> TvvfResult<Written>
where
    I: for<'a> FnMut(&'a mut dyn FnMut(u64) -> TvvfResult<()>) -> TvvfResult<()>,
    R: for<'a> FnMut(&'a mut dyn FnMut(&[u8]) -> TvvfResult<()>) -> TvvfResult<()>,
{
    let mut ids_checksum = 0xcbf2_9ce4_8422_2325;
    let mut seen = 0;
    ids(&mut |id| {
        ids_checksum = fnv1a64_update(ids_checksum, &id.to_le_bytes());
        seen += 1;
        Ok(())
    })?;
    if seen != rows {
        return Err(corrupt("tvvf writer id stream changed row count"));
    }
    let mut header = Map::new();
    header.insert(
        "schema_version".to_string(),
        Value::from(TVVF_SCHEMA_VERSION),
    );
    header.insert("dtype".to_string(), Value::from(dtype.as_str()));
    header.insert("dim".to_string(), Value::from(dim));
    header.insert("row_count".to_string(), Value::from(rows));
    header.insert("ids_checksum".to_string(), Value::from(ids_checksum));
    if magic == TVVF_DELTA_MAGIC {
        header.insert(
            "deleted_ids".to_string(),
            Value::Array(deleted.iter().copied().map(Value::from).collect()),
        );
    }
    let header = py_canonical_json(&Value::Object(header))
        .map_err(core_error)?
        .into_bytes();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_file_name(format!(
        "{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy()
    ));
    let written = (|| {
        // The per-row emit pattern would otherwise issue one write syscall per
        // id, checksum, and row: tens of millions at production scale, which
        // pathologically slow filesystems (network mounts, FUSE) cannot absorb.
        let mut file = io::BufWriter::with_capacity(4 << 20, File::create(&temporary)?);
        let mut sha = Sha256::new();
        let mut bytes = 0;
        hashed_write(&mut file, &mut sha, &mut bytes, magic)?;
        hashed_write(
            &mut file,
            &mut sha,
            &mut bytes,
            &(header.len() as u64).to_le_bytes(),
        )?;
        hashed_write(&mut file, &mut sha, &mut bytes, &header)?;
        let mut count = 0;
        ids(&mut |id| {
            count += 1;
            hashed_write(&mut file, &mut sha, &mut bytes, &id.to_le_bytes())
        })?;
        if count != rows {
            return Err(corrupt("tvvf writer id stream changed row count"));
        }
        let stride = row_stride(dtype, dim)?;
        let mut count = 0;
        payloads(&mut |row| {
            if row.len() != stride {
                return Err(corrupt("tvvf writer row has unexpected encoded length"));
            }
            count += 1;
            hashed_write(
                &mut file,
                &mut sha,
                &mut bytes,
                &row_checksum(row).to_le_bytes(),
            )
        })?;
        if count != rows {
            return Err(corrupt("tvvf writer checksum stream changed row count"));
        }
        let mut count = 0;
        payloads(&mut |row| {
            if row.len() != stride {
                return Err(corrupt("tvvf writer row has unexpected encoded length"));
            }
            count += 1;
            hashed_write(&mut file, &mut sha, &mut bytes, row)
        })?;
        if count != rows {
            return Err(corrupt("tvvf writer payload stream changed row count"));
        }
        let file = match file.into_inner() {
            Ok(file) => file,
            Err(error) => return Err(TvvfError::Io(error.into_error())),
        };
        if fsync {
            file.sync_all()?;
        }
        Ok(Written {
            sha256: format!("{:x}", sha.finalize()),
            file_bytes: bytes,
            ids_checksum,
        })
    })();
    let written = match written {
        Ok(value) => value,
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
    };
    fs::rename(&temporary, path)?;
    if fsync {
        if let Ok(parent) = File::open(path.parent().unwrap_or_else(|| Path::new("."))) {
            parent.sync_all()?;
        }
    }
    Ok(written)
}

fn hashed_write<W: Write>(
    file: &mut W,
    sha: &mut Sha256,
    bytes: &mut u64,
    data: &[u8],
) -> TvvfResult<()> {
    file.write_all(data)?;
    sha.update(data);
    *bytes = bytes
        .checked_add(data.len() as u64)
        .ok_or_else(|| invalid("tvvf file length overflow"))?;
    Ok(())
}

fn write_base_manifest(
    base: &Path,
    index_key: &str,
    vf_epoch: u64,
    dtype: TvvfDtype,
    dim: usize,
    rows: usize,
    written: Written,
    fsync: bool,
) -> TvvfResult<Value> {
    let manifest_file = delta_dir(base).join(TVVF_DELTA_MANIFEST_NAME);
    let next_seq = if manifest_file.is_file() {
        read_manifest(&manifest_file)?
            .as_object()
            .and_then(|object| object.get("next_seq"))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    } else {
        0
    };
    let manifest = serde_json::json!({
        "schema_version": TVVF_SCHEMA_VERSION,
        "index_key": index_key,
        "vf_epoch": vf_epoch,
        "base": {
            "file_name": base.file_name().unwrap_or_default().to_string_lossy(),
            "sha256": written.sha256,
            "file_bytes": written.file_bytes,
            "dtype": dtype.as_str(),
            "dim": dim,
            "row_count": rows,
            "ids_checksum": written.ids_checksum,
        },
        "deltas": [],
        "next_seq": next_seq,
    });
    write_manifest(&manifest_file, &manifest, fsync)?;
    Ok(manifest)
}

fn write_manifest(path: &Path, value: &Value, fsync: bool) -> TvvfResult<()> {
    write_pretty_json_atomic(path, value, fsync)
        .map(|_| ())
        .map_err(core_error)
}

fn read_manifest(path: &Path) -> TvvfResult<Value> {
    serde_json::from_slice(&fs::read(path)?)
        .map_err(|error| corrupt(format!("tvvf manifest is corrupt: {error}")))
}

pub(crate) fn validate_manifest_identity(
    manifest: &Value,
    expected_key: &str,
    expected_epoch: u64,
) -> TvvfResult<(String, u64)> {
    let object = json_object(manifest, "tvvf manifest")?;
    if object.get("schema_version").and_then(Value::as_u64) != Some(TVVF_SCHEMA_VERSION) {
        return Err(corrupt("unsupported tvvf manifest schema version"));
    }
    let key = object
        .get("index_key")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| corrupt("tvvf manifest is missing index_key"))?
        .to_string();
    let epoch = object
        .get("vf_epoch")
        .and_then(Value::as_u64)
        .ok_or_else(|| corrupt("tvvf manifest is missing vf_epoch"))?;
    if !expected_key.is_empty() && (key != expected_key || epoch != expected_epoch) {
        return Err(corrupt(
            "tvvf manifest index key or epoch does not match its path",
        ));
    }
    Ok((key, epoch))
}

fn validate_manifest_base(manifest: &Value, path: &Path, base: &SegmentIndex) -> TvvfResult<()> {
    let base_block = json_object(
        json_object(manifest, "tvvf manifest")?
            .get("base")
            .ok_or_else(|| corrupt("tvvf manifest is missing base"))?,
        "tvvf manifest base",
    )?;
    if base_block.get("file_name").and_then(Value::as_str)
        != path.file_name().and_then(|value| value.to_str())
        || base_block.get("dtype").and_then(Value::as_str) != Some(base.dtype.as_str())
        || base_block.get("dim").and_then(Value::as_u64) != Some(base.dim as u64)
        || base_block.get("row_count").and_then(Value::as_u64) != Some(base.ids.len() as u64)
    {
        return Err(corrupt("tvvf manifest base does not match base header"));
    }
    Ok(())
}

fn read_header(path: &Path, magic: &[u8; 8], delta: bool) -> TvvfResult<Header> {
    let file = File::open(path)?;
    read_header_from_file(&file, magic, delta)
}

fn read_header_from_file(file: &File, magic: &[u8; 8], delta: bool) -> TvvfResult<Header> {
    let length = file.metadata()?.len();
    let mut prefix = [0; 16];
    read_exact_at(&file, &mut prefix, 0)
        .map_err(|error| corrupt(format!("tvvf segment prefix is truncated: {error}")))?;
    if &prefix[..8] != magic {
        return Err(corrupt(format!(
            "not a tvvf {} segment",
            if delta { "delta" } else { "base" }
        )));
    }
    let header_len = u64::from_le_bytes(prefix[8..].try_into().expect("header length"));
    let header_stop = 16_u64
        .checked_add(header_len)
        .filter(|value| *value <= length && header_len <= 64 * 1024 * 1024)
        .ok_or_else(|| corrupt("tvvf header is truncated or too large"))?;
    let mut bytes = vec![0; header_len as usize];
    read_exact_at(&file, &mut bytes, 16)
        .map_err(|error| corrupt(format!("tvvf header is truncated: {error}")))?;
    let header: Value = serde_json::from_slice(&bytes)
        .map_err(|error| corrupt(format!("tvvf header is corrupt: {error}")))?;
    let object = json_object(&header, "tvvf segment header")?;
    if object.get("schema_version").and_then(Value::as_u64) != Some(TVVF_SCHEMA_VERSION) {
        return Err(corrupt("unsupported tvvf segment schema version"));
    }
    let dtype = parse_dtype(
        object
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| corrupt("tvvf segment is missing dtype"))?,
    )
    .map_err(|_| corrupt("tvvf segment has invalid dtype"))?;
    let dim = object
        .get("dim")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| corrupt("tvvf segment has invalid dim"))?;
    let row_count = object
        .get("row_count")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value <= u32::MAX as usize)
        .ok_or_else(|| corrupt("tvvf segment has invalid row_count"))?;
    let ids_checksum = object
        .get("ids_checksum")
        .and_then(Value::as_u64)
        .ok_or_else(|| corrupt("tvvf segment is missing ids_checksum"))?;
    let mut deleted_ids = if delta {
        object
            .get("deleted_ids")
            .and_then(Value::as_array)
            .ok_or_else(|| corrupt("tvvf delta is missing deleted_ids"))?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .ok_or_else(|| corrupt("tvvf delta has invalid deleted id"))
            })
            .collect::<TvvfResult<Vec<_>>>()?
    } else {
        Vec::new()
    };
    deleted_ids.sort_unstable();
    if deleted_ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(corrupt("tvvf delta has duplicate deleted ids"));
    }
    let row_stride = row_stride(dtype, dim).map_err(|_| corrupt("tvvf row stride overflows"))?;
    let ids_bytes = (row_count as u64)
        .checked_mul(8)
        .ok_or_else(|| corrupt("tvvf ids section overflows"))?;
    let checksums_offset = header_stop
        .checked_add(ids_bytes)
        .ok_or_else(|| corrupt("tvvf ids section overflows"))?;
    let rows_offset = checksums_offset
        .checked_add(
            (row_count as u64)
                .checked_mul(4)
                .ok_or_else(|| corrupt("tvvf checksum section overflows"))?,
        )
        .ok_or_else(|| corrupt("tvvf checksum section overflows"))?;
    let expected = rows_offset
        .checked_add(
            (row_count as u64)
                .checked_mul(row_stride as u64)
                .ok_or_else(|| corrupt("tvvf rows section overflows"))?,
        )
        .ok_or_else(|| corrupt("tvvf rows section overflows"))?;
    if expected != length {
        return Err(corrupt("tvvf segment has invalid length"));
    }
    Ok(Header {
        dtype,
        dim,
        row_count,
        ids_checksum,
        deleted_ids,
        ids_offset: header_stop,
        checksums_offset,
        rows_offset,
        row_stride,
    })
}

fn read_index(
    path: &Path,
    magic: &[u8; 8],
    delta: bool,
    expected_sha256: Option<&str>,
) -> TvvfResult<SegmentIndex> {
    let file = Arc::new(File::open(path)?);
    read_index_from_file(path, file, magic, delta, expected_sha256)
}

fn read_index_from_file(
    path: &Path,
    file: Arc<File>,
    magic: &[u8; 8],
    delta: bool,
    expected_sha256: Option<&str>,
) -> TvvfResult<SegmentIndex> {
    if let Some(expected) = expected_sha256 {
        if sha256_file_hex_from_file(&file)? != expected {
            return Err(corrupt(format!(
                "tvvf delta segment failed checksum: {}",
                path.file_name().unwrap_or_default().to_string_lossy()
            )));
        }
    }
    let header = read_header_from_file(&file, magic, delta)?;
    let mut ids = Vec::with_capacity(header.row_count);
    let mut hash = 0xcbf2_9ce4_8422_2325;
    let mut offset = header.ids_offset;
    let mut remaining = header.row_count;
    let mut buffer = vec![0; 64 * 1024];
    while remaining > 0 {
        let count = (remaining * 8).min(buffer.len());
        read_exact_at(&file, &mut buffer[..count], offset)
            .map_err(|error| corrupt(format!("tvvf ids section is truncated: {error}")))?;
        hash = fnv1a64_update(hash, &buffer[..count]);
        ids.extend(
            buffer[..count]
                .chunks_exact(8)
                .map(|value| u64::from_le_bytes(value.try_into().expect("u64"))),
        );
        offset += count as u64;
        remaining -= count / 8;
    }
    if hash != header.ids_checksum {
        return Err(corrupt("tvvf ids section failed checksum"));
    }
    let mut lookup_rows = (0..ids.len()).map(|value| value as u32).collect::<Vec<_>>();
    lookup_rows.sort_unstable_by_key(|row| ids[*row as usize]);
    if lookup_rows
        .windows(2)
        .any(|pair| ids[pair[0] as usize] == ids[pair[1] as usize])
    {
        return Err(corrupt("tvvf segment contains duplicate stable ids"));
    }
    if header.deleted_ids.iter().any(|id| {
        lookup_rows
            .binary_search_by_key(id, |row| ids[*row as usize])
            .is_ok()
    }) {
        return Err(corrupt("tvvf delta both upserts and deletes a stable id"));
    }
    Ok(SegmentIndex {
        path: path.to_path_buf(),
        file,
        dtype: header.dtype,
        dim: header.dim,
        row_stride: header.row_stride,
        ids,
        lookup_rows,
        deleted_ids: header.deleted_ids,
        checksums_offset: header.checksums_offset,
        rows_offset: header.rows_offset,
    })
}

fn validate_delta_segment(
    segment: &SegmentIndex,
    base: &SegmentIndex,
    name: &str,
) -> TvvfResult<()> {
    if segment.dtype != base.dtype || segment.dim != base.dim {
        return Err(corrupt(format!(
            "tvvf delta segment has incompatible dtype or dim: {name}"
        )));
    }
    Ok(())
}

fn coverage(base: &SegmentIndex, deltas: &[SegmentIndex]) -> (u64, u64, HashMap<u64, bool>) {
    let mut live = base.ids.len() as i64;
    let mut changed = HashMap::<u64, bool>::new();
    for segment in deltas {
        for id in &segment.deleted_ids {
            let was_live = changed
                .get(id)
                .copied()
                .unwrap_or_else(|| base.row_for(*id).is_some());
            if was_live {
                live -= 1;
            }
            changed.insert(*id, false);
        }
        for id in &segment.ids {
            let was_live = changed
                .get(id)
                .copied()
                .unwrap_or_else(|| base.row_for(*id).is_some());
            if !was_live {
                live += 1;
            }
            changed.insert(*id, true);
        }
    }
    (
        live.max(0) as u64,
        changed.values().filter(|value| !**value).count() as u64,
        changed,
    )
}

fn latest_rows(reader: &TvvfReader) -> HashMap<u64, Latest> {
    let mut latest = HashMap::new();
    for (segment, delta) in reader.deltas.iter().enumerate() {
        for id in &delta.deleted_ids {
            latest.insert(*id, Latest::Deleted);
        }
        for (row, id) in delta.ids.iter().copied().enumerate() {
            latest.insert(id, Latest::Row(segment, row));
        }
    }
    latest
}

fn fold_count(reader: &TvvfReader, latest: &HashMap<u64, Latest>) -> TvvfResult<usize> {
    let mut count = 0_usize;
    visit_fold_ids(reader, latest, &mut |_| {
        count = count
            .checked_add(1)
            .ok_or_else(|| invalid("tvvf fold row count overflow"))?;
        Ok(())
    })?;
    Ok(count)
}

fn visit_fold_ids(
    reader: &TvvfReader,
    latest: &HashMap<u64, Latest>,
    emit: &mut dyn FnMut(u64) -> TvvfResult<()>,
) -> TvvfResult<()> {
    for id in &reader.base.ids {
        if !latest.contains_key(id) {
            emit(*id)?;
        }
    }
    for (segment, delta) in reader.deltas.iter().enumerate() {
        for (row, id) in delta.ids.iter().copied().enumerate() {
            if latest.get(&id) == Some(&Latest::Row(segment, row)) {
                emit(id)?;
            }
        }
    }
    Ok(())
}

fn visit_fold_rows(
    reader: &TvvfReader,
    latest: &HashMap<u64, Latest>,
    emit: &mut dyn FnMut(&[u8]) -> TvvfResult<()>,
) -> TvvfResult<()> {
    visit_rows(&reader.base, |_, id| !latest.contains_key(&id), emit)?;
    for (segment, delta) in reader.deltas.iter().enumerate() {
        visit_rows(
            delta,
            |row, id| latest.get(&id) == Some(&Latest::Row(segment, row)),
            emit,
        )?;
    }
    Ok(())
}

fn visit_rows<F>(
    segment: &SegmentIndex,
    selected: F,
    emit: &mut dyn FnMut(&[u8]) -> TvvfResult<()>,
) -> TvvfResult<()>
where
    F: Fn(usize, u64) -> bool,
{
    let mut checksums = File::open(&segment.path)?;
    let mut payloads = File::open(&segment.path)?;
    checksums.seek(SeekFrom::Start(segment.checksums_offset))?;
    payloads.seek(SeekFrom::Start(segment.rows_offset))?;
    let mut checksum = [0; 4];
    let mut row = vec![0; segment.row_stride];
    for (index, id) in segment.ids.iter().copied().enumerate() {
        checksums.read_exact(&mut checksum)?;
        payloads.read_exact(&mut row)?;
        if selected(index, id) {
            if row_checksum(&row) != u32::from_le_bytes(checksum) {
                return Err(corrupt(format!(
                    "tvvf fold found corrupt source row for stable id {id}"
                )));
            }
            emit(&row)?;
        }
    }
    Ok(())
}

fn json_object<'a>(value: &'a Value, context: &str) -> TvvfResult<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| corrupt(format!("{context} must be an object")))
}

fn core_error(error: CoreError) -> TvvfError {
    corrupt(error.message())
}

/// Hashes a descriptor rather than reopening its pathname. The caller retains the
/// descriptor for the reader's lifetime, so validation and later serving always
/// refer to the same file even if epoch GC unlinks the name on Unix.
fn sha256_file_hex_from_file(file: &File) -> io::Result<String> {
    let mut copy = file.try_clone()?;
    copy.seek(SeekFrom::Start(0))?;
    let mut sha = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = copy.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        sha.update(&buffer[..read]);
    }
    Ok(format!("{:x}", sha.finalize()))
}

fn corrupt(message: impl Into<String>) -> TvvfError {
    TvvfError::Corrupt(message.into())
}

fn invalid(message: impl Into<String>) -> TvvfError {
    TvvfError::Invalid(message.into())
}

#[cfg(unix)]
fn read_exact_at(file: &File, bytes: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(bytes, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, bytes: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut read = 0;
    while read < bytes.len() {
        let count = file.seek_read(&mut bytes[read..], offset + read as u64)?;
        if count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short positional read",
            ));
        }
        read += count;
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn read_exact_at(file: &File, bytes: &mut [u8], offset: u64) -> io::Result<()> {
    let mut copy = file.try_clone()?;
    copy.seek(SeekFrom::Start(offset))?;
    copy.read_exact(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "lodedb_tvvf_{name}_{}_{}_{nonce}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed),
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn borrowed(rows: &[(u64, Vec<f32>)]) -> Vec<(u64, &[f32])> {
        rows.iter().map(|(id, row)| (*id, row.as_slice())).collect()
    }

    fn assert_rows(actual: &[Option<Vec<f32>>], expected: &[Option<Vec<f32>>], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            match (actual, expected) {
                (None, None) => {}
                (Some(actual), Some(expected)) => {
                    assert_eq!(actual.len(), expected.len());
                    for (actual, expected) in actual.iter().zip(expected) {
                        assert!(
                            (actual - expected).abs() <= tolerance,
                            "{actual} != {expected}"
                        );
                    }
                }
                _ => panic!("row presence mismatch"),
            }
        }
    }

    #[test]
    fn f16_encoder_matches_known_bit_patterns() {
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-2.5), 0xc100);
        assert_eq!(f32_to_f16_bits(65_504.0), 0x7bff);
        assert_eq!(f32_to_f16_bits(6.103_515_6e-5), 0x0400);
        assert_eq!(f32_to_f16_bits(5.960_464_5e-8), 0x0001);
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16_bits(f32::NEG_INFINITY), 0xfc00);
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::NAN)).is_nan());
        assert_eq!(f32_to_f16_bits(1.000_488_3), 0x3c00);
    }

    #[test]
    fn base_round_trips_all_precisions() {
        let source = vec![(11, vec![1.0, -2.5, 0.25]), (22, vec![0.0, 0.75, -0.5])];
        for dtype in [TvvfDtype::Float16, TvvfDtype::Float32, TvvfDtype::Int8] {
            let dir = temp_dir(dtype.as_str());
            let manifest = record_base(&dir, "index", 4, dtype, 3, &borrowed(&source)).unwrap();
            let reader = TvvfReader::open(&dir, &manifest).unwrap();
            let expected = source
                .iter()
                .map(|(_, row)| Some(decode_row(dtype, 3, &encode_row(dtype, 3, row).unwrap())))
                .collect::<Vec<_>>();
            assert_rows(&reader.fetch_rows(&[11, 22]), &expected, 1e-6);
            assert_eq!(reader.coverage(), (2, 0));
            assert_eq!(reader.index_key(), "index");
        }
    }

    #[test]
    fn delta_latest_wins_and_tombstones_hide_rows() {
        let dir = temp_dir("delta");
        let base = vec![(1, vec![1.0, 1.0]), (2, vec![2.0, 2.0])];
        let _ = record_base(&dir, "index", 0, TvvfDtype::Float32, 2, &borrowed(&base)).unwrap();
        let upserts = vec![(1, vec![9.0, 9.0]), (3, vec![3.0, 3.0])];
        let manifest = append_delta(&dir, "index", 0, &borrowed(&upserts), &[2]).unwrap();
        let reader = TvvfReader::open(&dir, &manifest).unwrap();
        assert_eq!(
            reader.fetch_rows(&[1, 2, 3, 999]),
            vec![Some(vec![9.0, 9.0]), None, Some(vec![3.0, 3.0]), None]
        );
        assert_eq!(reader.coverage(), (2, 1));
    }

    #[test]
    fn resident_reader_appends_only_the_new_delta_index() {
        let dir = temp_dir("resident_append");
        let base = vec![(1, vec![1.0, 1.0]), (2, vec![2.0, 2.0])];
        let manifest =
            record_base(&dir, "index", 0, TvvfDtype::Float32, 2, &borrowed(&base)).unwrap();
        let mut reader = TvvfReader::open(&dir, &manifest).unwrap();
        let upserts = vec![(1, vec![9.0, 9.0]), (3, vec![3.0, 3.0])];
        let manifest = append_delta(&dir, "index", 0, &borrowed(&upserts), &[2]).unwrap();
        let delta = load_latest_delta_segment(&dir, "index", 0, &manifest).unwrap();
        reader.append_delta_segment(delta).unwrap();

        assert_eq!(
            reader.fetch_rows(&[1, 2, 3]),
            vec![Some(vec![9.0, 9.0]), None, Some(vec![3.0, 3.0])]
        );
        assert_eq!(reader.coverage(), (2, 1));

        let next = vec![(2, vec![20.0, 20.0]), (4, vec![4.0, 4.0])];
        let manifest = append_delta(&dir, "index", 0, &borrowed(&next), &[3]).unwrap();
        let delta = load_latest_delta_segment(&dir, "index", 0, &manifest).unwrap();
        reader.append_delta_segment(delta).unwrap();
        assert_eq!(
            reader.fetch_rows(&[1, 2, 3, 4]),
            vec![
                Some(vec![9.0, 9.0]),
                Some(vec![20.0, 20.0]),
                None,
                Some(vec![4.0, 4.0]),
            ]
        );
        assert_eq!(reader.coverage(), (3, 1));
    }

    #[cfg(unix)]
    #[test]
    fn reader_fetches_base_and_delta_after_their_paths_are_unlinked() {
        let dir = temp_dir("unlinked_snapshot");
        let base = vec![(1, vec![1.0, 1.0]), (2, vec![2.0, 2.0])];
        record_base(
            &dir,
            "index",
            0,
            TvvfDtype::Float32,
            2,
            &borrowed(&base),
        )
        .unwrap();
        let updates = vec![(2, vec![20.0, 20.0]), (3, vec![3.0, 3.0])];
        let manifest = append_delta(&dir, "index", 0, &borrowed(&updates), &[]).unwrap();
        let delta_name = manifest["deltas"][0]["file_name"]
            .as_str()
            .expect("delta file name");
        let delta_checksum = manifest["deltas"][0]["sha256"]
            .as_str()
            .expect("delta checksum")
            .to_string();
        let reader = TvvfReader::open(&dir, &manifest).unwrap();

        let base_path = base_path(&dir, "index", 0);
        // Retain handles before unlinking, then run the same validation helper
        // TvvfReader::open and the incremental attach path use. This proves the
        // header, checksum, and ID validation never reopens the vanished name.
        let held_base = Arc::new(File::open(&base_path).unwrap());
        let delta_path = delta_dir(&base_path).join(delta_name);
        let held_delta = Arc::new(File::open(&delta_path).unwrap());
        fs::remove_file(&base_path).unwrap();
        fs::remove_file(&delta_path).unwrap();

        let validated_base =
            read_index_from_file(&base_path, held_base, TVVF_BASE_MAGIC, false, None).unwrap();
        let validated_delta = read_index_from_file(
            &delta_path,
            held_delta,
            TVVF_DELTA_MAGIC,
            true,
            Some(&delta_checksum),
        )
        .unwrap();
        assert_eq!(validated_base.ids, vec![1, 2]);
        assert_eq!(validated_delta.ids, vec![2, 3]);

        assert_eq!(
            reader.fetch_rows(&[1, 2, 3]),
            vec![
                Some(vec![1.0, 1.0]),
                Some(vec![20.0, 20.0]),
                Some(vec![3.0, 3.0]),
            ]
        );
        drop(reader);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn fold_matches_base_plus_multiple_deltas() {
        let dir = temp_dir("fold");
        let base = vec![
            (1, vec![1.0, 0.0]),
            (2, vec![2.0, 0.0]),
            (3, vec![3.0, 0.0]),
        ];
        let _ = record_base(&dir, "index", 8, TvvfDtype::Float16, 2, &borrowed(&base)).unwrap();
        let first = vec![(1, vec![10.0, 0.0]), (4, vec![4.0, 0.0])];
        let _ = append_delta(&dir, "index", 8, &borrowed(&first), &[2]).unwrap();
        let second = vec![(4, vec![40.0, 0.0]), (5, vec![5.0, 0.0])];
        let before_manifest = append_delta(&dir, "index", 8, &borrowed(&second), &[3]).unwrap();
        let before = TvvfReader::open(&dir, &before_manifest)
            .unwrap()
            .fetch_rows(&[1, 2, 3, 4, 5]);
        let folded = fold(&dir, "index", 8).unwrap();
        let after = TvvfReader::open(&dir, &folded)
            .unwrap()
            .fetch_rows(&[1, 2, 3, 4, 5]);
        assert_rows(&before, &after, 1e-6);
    }

    #[test]
    fn corrupt_payload_fails_open_but_bad_index_data_fails_open() {
        let rows = vec![(1, vec![1.0, 2.0]), (2, vec![3.0, 4.0])];
        let dir = temp_dir("corrupt_payload");
        let manifest =
            record_base(&dir, "index", 0, TvvfDtype::Float32, 2, &borrowed(&rows)).unwrap();
        let path = base_path(&dir, "index", 0);
        let header = read_header(&path, TVVF_BASE_MAGIC, false).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        bytes[header.rows_offset as usize] ^= 0xff;
        fs::write(&path, bytes).unwrap();
        let reader = TvvfReader::open(&dir, &manifest).unwrap();
        assert_eq!(reader.fetch_rows(&[1, 2]), vec![None, Some(vec![3.0, 4.0])]);
        assert_eq!(reader.corrupt_rows_seen(), 1);

        let truncated = temp_dir("truncated");
        let manifest = record_base(
            &truncated,
            "index",
            0,
            TvvfDtype::Float32,
            2,
            &borrowed(&rows),
        )
        .unwrap();
        let path = base_path(&truncated, "index", 0);
        let header = read_header(&path, TVVF_BASE_MAGIC, false).unwrap();
        fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(header.checksums_offset - 1)
            .unwrap();
        match TvvfReader::open(&truncated, &manifest) {
            Err(error) => assert!(error.is_corrupt()),
            Ok(_) => panic!("truncated ids section opened"),
        }

        let bad_magic = temp_dir("bad_magic");
        let manifest = record_base(
            &bad_magic,
            "index",
            0,
            TvvfDtype::Float32,
            2,
            &borrowed(&rows),
        )
        .unwrap();
        let path = base_path(&bad_magic, "index", 0);
        let mut bytes = fs::read(&path).unwrap();
        bytes[0] = b'X';
        fs::write(&path, bytes).unwrap();
        match TvvfReader::open(&bad_magic, &manifest) {
            Err(error) => assert!(error.is_corrupt()),
            Ok(_) => panic!("bad magic opened"),
        }
    }

    #[test]
    fn several_thousand_rows_match_a_btreemap_model() {
        let dir = temp_dir("property");
        let dim = 96;
        let base = (0..2400_u64)
            .map(|id| (id, generated_row(id, dim, 1)))
            .collect::<Vec<_>>();
        let mut model = base.iter().cloned().collect::<BTreeMap<_, _>>();
        let _ = record_base(&dir, "index", 1, TvvfDtype::Float32, dim, &borrowed(&base)).unwrap();
        for segment in 0..3_u64 {
            let updates = (0..400_u64)
                .map(|offset| {
                    let id = offset + segment * 83;
                    (id, generated_row(id, dim, segment + 10))
                })
                .collect::<Vec<_>>();
            let deleted = (0..50_u64)
                .map(|offset| 800 + segment * 50 + offset)
                .collect::<Vec<_>>();
            for (id, row) in &updates {
                model.insert(*id, row.clone());
            }
            for id in &deleted {
                model.remove(id);
            }
            let _ = append_delta(&dir, "index", 1, &borrowed(&updates), &deleted).unwrap();
        }
        let manifest = read_manifest(&manifest_path(&dir, "index", 1)).unwrap();
        let reader = TvvfReader::open(&dir, &manifest).unwrap();
        let ids = (0..700_u64)
            .map(|offset| offset * 37 % 2700)
            .collect::<Vec<_>>();
        let expected = ids
            .iter()
            .map(|id| model.get(id).cloned())
            .collect::<Vec<_>>();
        assert_rows(&reader.fetch_rows(&ids), &expected, 0.0);
    }

    #[test]
    fn scattered_single_segment_fetch_matches_a_btreemap_model() {
        let dir = temp_dir("scattered_single_segment");
        let dim = 12;
        let base = (0..4096_u64)
            .map(|id| (id, generated_row(id, dim, 7)))
            .collect::<Vec<_>>();
        let model = base.iter().cloned().collect::<BTreeMap<_, _>>();
        let manifest =
            record_base(&dir, "index", 0, TvvfDtype::Float32, dim, &borrowed(&base)).unwrap();
        let reader = TvvfReader::open(&dir, &manifest).unwrap();

        // Every seventh row yields hundreds of separate runs, exceeding the
        // per-fetch task limit and exercising multiple parallel run chunks.
        let mut ids = (0..384_u64).map(|offset| offset * 7).collect::<Vec<_>>();
        ids.extend([7, 777, 2_681]);
        let expected = ids
            .iter()
            .map(|id| model.get(id).cloned())
            .collect::<Vec<_>>();
        assert_rows(&reader.fetch_rows(&ids), &expected, 0.0);
    }

    #[test]
    fn concurrent_fetch_smoke_uses_positional_reads() {
        let dir = temp_dir("concurrent");
        let rows = (0..512_u64)
            .map(|id| (id, generated_row(id, 12, 3)))
            .collect::<Vec<_>>();
        let manifest =
            record_base(&dir, "index", 0, TvvfDtype::Float32, 12, &borrowed(&rows)).unwrap();
        let reader = Arc::new(TvvfReader::open(&dir, &manifest).unwrap());
        (0..16_usize).into_par_iter().for_each(|task| {
            let ids = (0..128_u64)
                .map(|offset| (offset + task as u64 * 13) % 512)
                .collect::<Vec<_>>();
            assert!(reader.fetch_rows(&ids).iter().all(Option::is_some));
        });
    }

    #[test]
    fn record_base_rejects_an_empty_index_key() {
        let dir = temp_dir("empty_key");
        let rows = vec![(1_u64, generated_row(1, 4, 1))];
        let error = record_base(&dir, "", 0, TvvfDtype::Float32, 4, &borrowed(&rows)).unwrap_err();
        assert!(!error.is_corrupt());
        assert!(!base_path(&dir, "", 0).exists());
    }

    #[test]
    fn fold_rejects_a_manifest_with_a_foreign_identity() {
        let dir = temp_dir("fold_identity");
        let rows = (0..8_u64)
            .map(|id| (id, generated_row(id, 4, 5)))
            .collect::<Vec<_>>();
        record_base(&dir, "index", 0, TvvfDtype::Float32, 4, &borrowed(&rows)).unwrap();
        let path = manifest_path(&dir, "index", 0);
        let mut manifest = read_manifest(&path).unwrap();
        manifest
            .as_object_mut()
            .unwrap()
            .insert("vf_epoch".to_string(), Value::from(9_u64));
        write_manifest(&path, &manifest, false).unwrap();
        let error = fold(&dir, "index", 0).unwrap_err();
        assert!(error.is_corrupt());
        assert!(!base_path(&dir, "index", 1).exists());
    }

    #[test]
    fn append_delta_rejects_an_exhausted_sequence() {
        let dir = temp_dir("seq_overflow");
        let rows = (0..8_u64)
            .map(|id| (id, generated_row(id, 4, 9)))
            .collect::<Vec<_>>();
        record_base(&dir, "index", 0, TvvfDtype::Float32, 4, &borrowed(&rows)).unwrap();
        let path = manifest_path(&dir, "index", 0);
        let mut manifest = read_manifest(&path).unwrap();
        manifest
            .as_object_mut()
            .unwrap()
            .insert("next_seq".to_string(), Value::from(u64::MAX));
        write_manifest(&path, &manifest, false).unwrap();
        let upserts = vec![(1_u64, generated_row(1, 4, 9))];
        let error = append_delta(&dir, "index", 0, &borrowed(&upserts), &[]).unwrap_err();
        assert!(error.is_corrupt());
    }

    fn generated_row(id: u64, dim: usize, salt: u64) -> Vec<f32> {
        (0..dim)
            .map(|column| ((id.wrapping_mul(31) + column as u64 * 17 + salt) % 997) as f32 / 997.0)
            .collect()
    }
}
