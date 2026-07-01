//! Durable, process-shared log-sequence-number (LSN) allocator.
//!
//! Concurrent WAL appenders need LSNs that are unique and monotonic across
//! independent processes, which the per-handle in-memory generation counter
//! cannot provide (each handle would hand out the same numbers). This allocator
//! persists the last handed-out LSN in a small `<index_key>.lsn` file and
//! reserves ranges under a short exclusive OS lock on that file, so concurrent
//! reservers never collide. It is the shared ordering source for the concurrent
//! append path tracked in issue #50; the single-writer path keeps using the
//! in-memory generation as its LSN and does not touch this file.
//!
//! The lock is held only for the read-modify-write, not for a handle's lifetime,
//! and (like the writer lock) is released by the OS when the process exits, so a
//! crash mid-reservation never wedges the counter.

use crate::storage::util::{corrupt, CoreResult};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const LSN_SUFFIX: &str = ".lsn";
const LSN_MAGIC: &[u8; 8] = b"EELLSN01";
const LSN_RECORD_LEN: usize = LSN_MAGIC.len() + 8;

/// Returns the counter-file path for an index.
pub fn lsn_path(persistence_dir: &Path, index_key: &str) -> PathBuf {
    persistence_dir.join(format!("{index_key}{LSN_SUFFIX}"))
}

/// Reserves `count` contiguous LSNs and returns the first of the range
/// (`start ..= start + count - 1`).
///
/// `floor` is the highest LSN already durable in the store (the maximum of the
/// committed watermark and any WAL tail) and is authoritative: the counter is a
/// rebuildable optimization over it. The reservation never hands out an LSN at
/// or below `floor`, so a fresh, missing, or crash-torn counter seeds safely and
/// never reuses a sequence number. `count` must be non-zero. The read-modify-
/// write runs under a short exclusive lock on the counter file, so concurrent
/// reservers cannot collide or observe a partial write.
pub fn reserve(
    persistence_dir: &Path,
    index_key: &str,
    count: u64,
    floor: u64,
    fsync: bool,
) -> CoreResult<u64> {
    if count == 0 {
        return Err(corrupt("LSN reservation count must be non-zero"));
    }
    let path = lsn_path(persistence_dir, index_key);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| corrupt(format!("LSN counter directory could not be created: {error}")))?;
    }
    let mut file = open_exclusive(&path)?;
    let current = read_value(&mut file)?.unwrap_or(0).max(floor);
    let next = current
        .checked_add(count)
        .ok_or_else(|| corrupt("LSN counter would overflow u64"))?;
    write_value(&mut file, next, fsync)?;
    // Dropping `file` releases the OS lock and closes the handle.
    Ok(current + 1)
}

/// Reads the highest LSN the counter has handed out, or `None` when the counter
/// does not exist yet. Does not take the exclusive lock, so callers that need a
/// consistent value against concurrent reservers must serialize themselves; it
/// exists for read-only inspection and recovery seeding.
pub fn peek(persistence_dir: &Path, index_key: &str) -> CoreResult<Option<u64>> {
    let path = lsn_path(persistence_dir, index_key);
    if !path.is_file() {
        return Ok(None);
    }
    let mut file = OpenOptions::new()
        .read(true)
        .open(&path)
        .map_err(|error| corrupt(format!("LSN counter could not be opened: {error}")))?;
    read_value(&mut file)
}

fn read_value(file: &mut File) -> CoreResult<Option<u64>> {
    file.seek(SeekFrom::Start(0))
        .map_err(|error| corrupt(format!("LSN counter could not be read: {error}")))?;
    let mut buf = Vec::with_capacity(LSN_RECORD_LEN);
    file.read_to_end(&mut buf)
        .map_err(|error| corrupt(format!("LSN counter could not be read: {error}")))?;
    // The counter is a rebuildable optimization over the caller's floor, not a
    // source of truth, so it self-heals rather than failing closed. An empty
    // file (just created) or one left torn by a crash mid-write (short, or wrong
    // magic) reads as "no value", and `reserve` reseeds from the floor instead of
    // wedging. A structurally intact but partially rewritten value is still
    // clamped up to the floor by `reserve`, so it can never fall below the
    // store's durable maximum. A genuine read (I/O) error still propagates.
    if buf.len() != LSN_RECORD_LEN || &buf[..LSN_MAGIC.len()] != LSN_MAGIC {
        return Ok(None);
    }
    let mut value = [0_u8; 8];
    value.copy_from_slice(&buf[LSN_MAGIC.len()..]);
    Ok(Some(u64::from_be_bytes(value)))
}

fn write_value(file: &mut File, value: u64, fsync: bool) -> CoreResult<()> {
    file.seek(SeekFrom::Start(0))
        .map_err(|error| corrupt(format!("LSN counter could not be written: {error}")))?;
    file.write_all(LSN_MAGIC)
        .and_then(|_| file.write_all(&value.to_be_bytes()))
        .and_then(|_| file.flush())
        .map_err(|error| corrupt(format!("LSN counter could not be written: {error}")))?;
    // The record is fixed width, but truncate defensively so a shorter rewrite
    // could never leave trailing bytes from a longer prior write.
    file.set_len(LSN_RECORD_LEN as u64)
        .map_err(|error| corrupt(format!("LSN counter could not be sized: {error}")))?;
    if fsync {
        file.sync_all()
            .map_err(|error| corrupt(format!("LSN counter could not be fsynced: {error}")))?;
    }
    Ok(())
}

/// Opens the counter file with an exclusive hold for the read-modify-write.
///
/// Unix takes a blocking BSD advisory lock; Windows opens with an exclusive
/// share mode and retries while another reserver holds it. Both release when the
/// returned handle drops (or the process exits).
#[cfg(unix)]
fn open_exclusive(path: &Path) -> CoreResult<File> {
    use rustix::fs::{flock, FlockOperation};
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|error| corrupt(format!("LSN counter could not be opened: {error}")))?;
    flock(&file, FlockOperation::LockExclusive)
        .map_err(|error| corrupt(format!("LSN counter could not be locked: {error}")))?;
    Ok(file)
}

#[cfg(windows)]
fn open_exclusive(path: &Path) -> CoreResult<File> {
    use std::os::windows::fs::OpenOptionsExt;
    use std::time::{Duration, Instant};
    const ERROR_SHARING_VIOLATION: i32 = 32;
    const ERROR_LOCK_VIOLATION: i32 = 33;
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .share_mode(0)
            .open(path)
        {
            Ok(file) => return Ok(file),
            Err(error)
                if matches!(
                    error.raw_os_error(),
                    Some(ERROR_SHARING_VIOLATION) | Some(ERROR_LOCK_VIOLATION)
                ) =>
            {
                if Instant::now() >= deadline {
                    return Err(corrupt("LSN counter is locked by another reserver"));
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(error) => {
                return Err(corrupt(format!("LSN counter could not be opened: {error}")))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(tag: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "lodedb-lsn-{tag}-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn reserves_sequentially_from_one() {
        let dir = temp_dir("sequential");
        assert_eq!(reserve(&dir, "k", 1, 0, false).unwrap(), 1);
        assert_eq!(reserve(&dir, "k", 1, 0, false).unwrap(), 2);
        assert_eq!(reserve(&dir, "k", 1, 0, false).unwrap(), 3);
        assert_eq!(peek(&dir, "k").unwrap(), Some(3));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn reserves_contiguous_ranges() {
        let dir = temp_dir("ranges");
        assert_eq!(reserve(&dir, "k", 5, 0, false).unwrap(), 1);
        // The five-LSN reservation consumed 1..=5, so the next starts at 6.
        assert_eq!(reserve(&dir, "k", 1, 0, false).unwrap(), 6);
        assert_eq!(peek(&dir, "k").unwrap(), Some(6));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn floor_seeds_a_fresh_counter_without_reuse() {
        let dir = temp_dir("floor");
        // A store already durable up to LSN 10 must never hand out <= 10.
        assert_eq!(reserve(&dir, "k", 1, 10, false).unwrap(), 11);
        assert_eq!(reserve(&dir, "k", 1, 10, false).unwrap(), 12);
        // A stale (lower) floor cannot pull the counter backwards.
        assert_eq!(reserve(&dir, "k", 1, 3, false).unwrap(), 13);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn peek_is_none_before_first_reservation() {
        let dir = temp_dir("peek-none");
        assert_eq!(peek(&dir, "k").unwrap(), None);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_zero_count() {
        let dir = temp_dir("zero");
        assert!(reserve(&dir, "k", 0, 0, false).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn corrupt_counter_reseeds_from_floor() {
        let dir = temp_dir("corrupt");
        reserve(&dir, "k", 1, 0, false).unwrap();
        // A crash mid-write can leave the counter torn. Because the floor (the
        // store's durable max LSN) is authoritative, a garbage counter must not
        // wedge reservation: it reads as absent and the next reservation seeds
        // from the floor, then heals the file for later reservations.
        std::fs::write(lsn_path(&dir, "k"), b"not-a-counter").unwrap();
        assert_eq!(peek(&dir, "k").unwrap(), None);
        assert_eq!(reserve(&dir, "k", 1, 10, false).unwrap(), 11);
        assert_eq!(reserve(&dir, "k", 1, 0, false).unwrap(), 12);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn concurrent_reservations_are_unique_and_contiguous() {
        let dir = temp_dir("concurrent");
        let threads = 8_u64;
        let per_thread = 100_u64;
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let dir = dir.clone();
                std::thread::spawn(move || {
                    (0..per_thread)
                        .map(|_| reserve(&dir, "k", 1, 0, false).unwrap())
                        .collect::<Vec<u64>>()
                })
            })
            .collect();
        let mut all: Vec<u64> = handles
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect();
        all.sort_unstable();
        // Every LSN in 1..=threads*per_thread handed out exactly once: no gaps,
        // no duplicates, which is only possible if the lock serialized the
        // read-modify-write across all threads.
        let expected: Vec<u64> = (1..=threads * per_thread).collect();
        assert_eq!(all, expected);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
