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
//! Alongside the LSN, the counter carries a WAL *watermark*: the byte length of
//! the WAL's valid frame prefix as of the last append that recorded it. Because
//! every append advances it under the same lock, an appender can repair a
//! crashed peer's torn tail in O(1) (compare the file length to the watermark)
//! instead of rescanning the whole WAL on every append. The watermark is a
//! rebuildable hint over a full scan, not a source of truth: a v1 counter or a
//! torn value simply reads as "unknown" and the appender falls back to a scan.
//!
//! The lock is held only for the read-modify-write, not for a handle's lifetime,
//! and (like the writer lock) is released by the OS when the process exits, so a
//! crash mid-reservation never wedges the counter.

use crate::storage::util::{corrupt, CoreResult};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const LSN_SUFFIX: &str = ".lsn";
// v1 stored only the LSN (magic + u64). v2 appends the WAL watermark and a CRC
// over the payload (magic + lsn + wal_len + crc32). Both are read back; every
// write is v2. The CRC matters because the record is fixed width and rewritten in
// place, so a crash mid-write can leave a full-length record with valid magic but
// a torn lsn/wal_len; the CRC catches that so it reads as absent rather than as a
// bogus watermark. Bumping the magic also keeps a shorter record from being
// misread across layouts.
const LSN_MAGIC_V1: &[u8; 8] = b"EELLSN01";
const LSN_MAGIC_V2: &[u8; 8] = b"EELLSN02";
const LSN_RECORD_LEN_V1: usize = LSN_MAGIC_V1.len() + 8;
// magic + lsn + wal_len + crc32.
const LSN_PAYLOAD_LEN: usize = 16;
const LSN_RECORD_LEN_V2: usize = LSN_MAGIC_V2.len() + LSN_PAYLOAD_LEN + 4;
// Stored in the watermark slot when the writer does not track the WAL length
// (the standalone `reserve` path). A real WAL can never be u64::MAX bytes, so it
// is an unambiguous "no watermark recorded" and reads back as `None`.
const WAL_LEN_UNKNOWN: u64 = u64::MAX;

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
    with_reserved(persistence_dir, index_key, count, floor, fsync, Ok)
}

/// Reserves `count` LSNs and runs `body` with the first of the range while the
/// counter lock is still held, returning whatever `body` returns.
///
/// The counter is advanced and persisted before `body` runs, so the LSNs are
/// spent even if `body` fails (an un-written tail simply becomes an LSN gap,
/// never a reuse). Holding the lock across `body` is what lets a concurrent WAL
/// append keep file order aligned with LSN order: an appender reserves and
/// writes its frames as one critical section, so a second appender cannot slot
/// a lower-numbered frame in between. `body` must therefore be short (a WAL
/// append), never a checkpoint or other long operation. See [`reserve`] for the
/// argument contract.
pub fn with_reserved<T>(
    persistence_dir: &Path,
    index_key: &str,
    count: u64,
    floor: u64,
    fsync: bool,
    body: impl FnOnce(u64) -> CoreResult<T>,
) -> CoreResult<T> {
    if count == 0 {
        return Err(corrupt("LSN reservation count must be non-zero"));
    }
    with_lock(persistence_dir, index_key, |file| {
        let current = read_counter(file)?
            .map(|counter| counter.lsn)
            .unwrap_or(0)
            .max(floor);
        let next = current
            .checked_add(count)
            .ok_or_else(|| corrupt("LSN counter would overflow u64"))?;
        // The standalone reserve path does not track the WAL, so it records no
        // watermark; the appender path uses `write_counter` directly to store one.
        write_counter(file, next, None, fsync)?;
        body(current + 1)
    })
}

/// Runs `body` while holding the counter's exclusive lock, handing it the open
/// counter file. This is the append serialization point: [`with_reserved`] takes
/// it to reserve LSNs and write WAL frames, and an appender's open takes it to
/// inspect and repair the WAL without racing a concurrent append. `body` must be
/// short (a counter update or a WAL append/repair), never a checkpoint.
pub fn with_lock<T>(
    persistence_dir: &Path,
    index_key: &str,
    body: impl FnOnce(&mut File) -> CoreResult<T>,
) -> CoreResult<T> {
    let path = lsn_path(persistence_dir, index_key);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            corrupt(format!("LSN counter directory could not be created: {error}"))
        })?;
    }
    let mut file = open_exclusive(&path)?;
    let result = body(&mut file);
    // Hold the lock until `body` returns, then release it by dropping the handle.
    drop(file);
    result
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
    Ok(read_counter(&mut file)?.map(|counter| counter.lsn))
}

/// A decoded counter record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Counter {
    /// Highest LSN handed out.
    pub lsn: u64,
    /// Byte length of the WAL's valid frame prefix as of the append that wrote
    /// this record, or `None` when unknown (a v1 counter, or the `reserve` path
    /// that does not track the WAL). `None` makes the appender fall back to a
    /// full WAL scan for that one repair.
    pub wal_len: Option<u64>,
}

/// Reads the counter, or `None` when it does not exist yet or is unreadable.
///
/// The counter is a rebuildable optimization over the caller's floor and WAL, not
/// a source of truth, so it self-heals rather than failing closed. An empty file
/// (just created) or one left torn by a crash mid-write (short, or wrong magic)
/// reads as `None`, and callers reseed the LSN from the floor and the watermark
/// from a full scan instead of wedging. A structurally intact but partially
/// rewritten value is still clamped up to the floor by `reserve`, so it can never
/// fall below the store's durable maximum. A genuine read (I/O) error still
/// propagates. Both the v1 (LSN-only) and v2 (LSN + watermark) layouts are read.
pub fn read_counter(file: &mut File) -> CoreResult<Option<Counter>> {
    file.seek(SeekFrom::Start(0))
        .map_err(|error| corrupt(format!("LSN counter could not be read: {error}")))?;
    let mut buf = Vec::with_capacity(LSN_RECORD_LEN_V2);
    file.read_to_end(&mut buf)
        .map_err(|error| corrupt(format!("LSN counter could not be read: {error}")))?;
    if buf.len() == LSN_RECORD_LEN_V2 && &buf[..LSN_MAGIC_V2.len()] == LSN_MAGIC_V2 {
        let payload = &buf[LSN_MAGIC_V2.len()..LSN_MAGIC_V2.len() + LSN_PAYLOAD_LEN];
        let stored_crc = read_u32(&buf[LSN_MAGIC_V2.len() + LSN_PAYLOAD_LEN..]);
        // The record is fixed width and rewritten in place, so a crash mid-write
        // can leave the length and magic intact but the lsn/wal_len torn (a mix of
        // old and new bytes). The CRC over the payload catches that: a torn counter
        // reads as absent, so the caller reseeds the LSN from the floor and the
        // watermark from a full scan rather than trusting a bogus value to truncate
        // the WAL at a non-frame boundary. Same 1-in-2^32 guarantee the WAL frames
        // rely on.
        if crate::storage::wal::crc32(payload) != stored_crc {
            return Ok(None);
        }
        let lsn = read_u64(&payload[..8]);
        let raw = read_u64(&payload[8..]);
        let wal_len = (raw != WAL_LEN_UNKNOWN).then_some(raw);
        Ok(Some(Counter { lsn, wal_len }))
    } else if buf.len() == LSN_RECORD_LEN_V1 && &buf[..LSN_MAGIC_V1.len()] == LSN_MAGIC_V1 {
        // Legacy LSN-only record. No CRC, but a torn lsn is benign: the caller
        // clamps it up to the floor, and the absent watermark forces a full scan.
        Ok(Some(Counter {
            lsn: read_u64(&buf[8..16]),
            wal_len: None,
        }))
    } else {
        Ok(None)
    }
}

/// Writes the counter in the v2 layout. `wal_len` is `None` when the writer does
/// not track the WAL (stored as the `WAL_LEN_UNKNOWN` sentinel). The watermark
/// must be persisted only after the WAL frame it describes is durable, so a crash
/// between the two leaves the watermark behind the frame and the next appender
/// drops the frame as an unacknowledged tail.
pub fn write_counter(
    file: &mut File,
    lsn: u64,
    wal_len: Option<u64>,
    fsync: bool,
) -> CoreResult<()> {
    let mut payload = [0_u8; LSN_PAYLOAD_LEN];
    payload[..8].copy_from_slice(&lsn.to_be_bytes());
    payload[8..].copy_from_slice(&wal_len.unwrap_or(WAL_LEN_UNKNOWN).to_be_bytes());
    let crc = crate::storage::wal::crc32(&payload);
    file.seek(SeekFrom::Start(0))
        .map_err(|error| corrupt(format!("LSN counter could not be written: {error}")))?;
    file.write_all(LSN_MAGIC_V2)
        .and_then(|_| file.write_all(&payload))
        .and_then(|_| file.write_all(&crc.to_be_bytes()))
        .and_then(|_| file.flush())
        .map_err(|error| corrupt(format!("LSN counter could not be written: {error}")))?;
    // The record is fixed width, but truncate defensively so a shorter rewrite
    // (e.g. over a legacy v1 record) could never leave trailing bytes behind.
    file.set_len(LSN_RECORD_LEN_V2 as u64)
        .map_err(|error| corrupt(format!("LSN counter could not be sized: {error}")))?;
    if fsync {
        file.sync_all()
            .map_err(|error| corrupt(format!("LSN counter could not be fsynced: {error}")))?;
    }
    Ok(())
}

fn read_u64(bytes: &[u8]) -> u64 {
    let mut value = [0_u8; 8];
    value.copy_from_slice(bytes);
    u64::from_be_bytes(value)
}

fn read_u32(bytes: &[u8]) -> u32 {
    let mut value = [0_u8; 4];
    value.copy_from_slice(bytes);
    u32::from_be_bytes(value)
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

    #[test]
    fn watermark_round_trips_and_unknown_reads_as_none() {
        let dir = temp_dir("watermark");
        let path = lsn_path(&dir, "k");
        {
            let mut file = open_exclusive(&path).unwrap();
            write_counter(&mut file, 7, Some(4096), false).unwrap();
        }
        {
            let mut file = open_exclusive(&path).unwrap();
            assert_eq!(
                read_counter(&mut file).unwrap(),
                Some(Counter {
                    lsn: 7,
                    wal_len: Some(4096),
                })
            );
        }
        // A writer that does not track the WAL records no watermark, which reads
        // back as `None` rather than as a zero-length WAL.
        {
            let mut file = open_exclusive(&path).unwrap();
            write_counter(&mut file, 9, None, false).unwrap();
        }
        {
            let mut file = open_exclusive(&path).unwrap();
            assert_eq!(
                read_counter(&mut file).unwrap(),
                Some(Counter {
                    lsn: 9,
                    wal_len: None,
                })
            );
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn reads_a_legacy_v1_counter_and_upgrades_it() {
        let dir = temp_dir("v1-compat");
        let path = lsn_path(&dir, "k");
        // A counter written by the pre-watermark format: magic + lsn, no watermark.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(LSN_MAGIC_V1);
        bytes.extend_from_slice(&5_u64.to_be_bytes());
        std::fs::write(&path, bytes).unwrap();
        {
            let mut file = open_exclusive(&path).unwrap();
            assert_eq!(
                read_counter(&mut file).unwrap(),
                Some(Counter {
                    lsn: 5,
                    wal_len: None,
                })
            );
        }
        // The next reservation preserves the LSN and rewrites the file as v2.
        assert_eq!(reserve(&dir, "k", 1, 0, false).unwrap(), 6);
        assert_eq!(peek(&dir, "k").unwrap(), Some(6));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn torn_payload_with_valid_magic_reads_as_none() {
        let dir = temp_dir("torn-payload");
        let path = lsn_path(&dir, "k");
        {
            let mut file = open_exclusive(&path).unwrap();
            write_counter(&mut file, 7, Some(4096), false).unwrap();
        }
        // A crash mid-rewrite can leave the fixed-width record full length with
        // valid magic but a torn lsn/wal_len. Flip a payload byte so the stored CRC
        // no longer matches: only the CRC distinguishes this from an intact record,
        // and it must read as absent so the appender rescans instead of truncating
        // the WAL to a bogus watermark.
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[10] ^= 0xFF;
        std::fs::write(&path, bytes).unwrap();
        {
            let mut file = open_exclusive(&path).unwrap();
            assert_eq!(read_counter(&mut file).unwrap(), None);
        }
        // Reservation reseeds from the floor and heals the file back to a valid v2.
        assert_eq!(reserve(&dir, "k", 1, 10, false).unwrap(), 11);
        assert_eq!(peek(&dir, "k").unwrap(), Some(11));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
