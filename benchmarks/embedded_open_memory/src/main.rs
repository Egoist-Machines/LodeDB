//! Memory probe for the embedded read path: a read-only open, an optional
//! `refresh()`, a first search and a second (warm) search over one store.
//!
//! For each stage it prints the process phys_footprint (what jetsam counts on
//! iOS), the lifetime peak footprint so far, live heap bytes and the heap peak
//! inside that stage (from a counting allocator), and the stage's wall time.
//! The last line is a digest of the first search's hits (ids and score bits),
//! so two builds can be checked for identical results.
//!
//! ```text
//! cargo run --release --manifest-path benchmarks/embedded_open_memory/Cargo.toml -- \
//!     <store-dir> [--no-refresh] [--mode vector|hybrid|lexical] [--filter] [--k 60] [--hold 30]
//! ```
//!
//! `run.sh` builds this probe against the working tree and against a base git
//! revision and runs both over the same stores. macOS only.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use lodedb_core::engine::CoreEngine;
use lodedb_core::types::CoreOpenOptions;
use serde_json::json;

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            grow(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc_zeroed(layout);
        if !ptr.is_null() {
            grow(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = System.realloc(ptr, layout, new_size);
        if !new.is_null() {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            grow(new_size);
        }
        new
    }
}

fn grow(size: usize) {
    let live = LIVE.fetch_add(size, Ordering::Relaxed) + size;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// (phys_footprint, lifetime max phys_footprint) in bytes.
fn footprint() -> (u64, u64) {
    let mut info: libc::rusage_info_v4 = unsafe { std::mem::zeroed() };
    let status = unsafe {
        libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            (&mut info as *mut libc::rusage_info_v4).cast(),
        )
    };
    if status != 0 {
        return (0, 0);
    }
    (info.ri_phys_footprint, info.ri_lifetime_max_phys_footprint)
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / 1_048_576.0
}

struct Probe {
    started: Instant,
}

impl Probe {
    fn start() -> Self {
        PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
        Self {
            started: Instant::now(),
        }
    }

    fn report(self, stage: &str) {
        let elapsed = self.started.elapsed();
        let (now, peak) = footprint();
        println!(
            "{stage:<8} {:>8.1} ms  footprint {:>7.1} MiB  peak {:>7.1} MiB  heap {:>7.1} MiB  heap-peak {:>7.1} MiB",
            elapsed.as_secs_f64() * 1000.0,
            mib(now),
            mib(peak),
            mib(LIVE.load(Ordering::Relaxed) as u64),
            mib(PEAK.load(Ordering::Relaxed) as u64),
        );
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut store = None;
    let mut refresh = true;
    let mut mode = "vector".to_string();
    let mut filter = false;
    let mut top_k = 60usize;
    let mut hold_secs = 0u64;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--no-refresh" => refresh = false,
            "--mode" => mode = args.next().expect("--mode needs a value"),
            "--filter" => filter = true,
            "--k" => {
                top_k = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--k needs a number")
            }
            "--hold" => {
                hold_secs = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--hold needs seconds")
            }
            other if store.is_none() => store = Some(other.to_string()),
            other => panic!("unexpected argument {other}"),
        }
    }
    let store = store.expect("usage: embedded-open-memory <store-dir> [options]");

    Probe::start().report("idle");

    let probe = Probe::start();
    let options = CoreOpenOptions {
        path: store.clone(),
        read_only: true,
        durability: "fsync".to_string(),
        commit_mode: "wal".to_string(),
        store_text: true,
        index_text: true,
        compress_text: true,
        chunk_character_limit: 8192,
        acquire_writer_lock: true,
    };
    let mut engine = CoreEngine::open_readonly(&store, options).expect("open_readonly");
    probe.report("open");

    if refresh {
        let probe = Probe::start();
        engine.refresh().expect("refresh");
        probe.report("refresh");
    }

    let index_id = engine
        .index_ids()
        .into_iter()
        .next()
        .expect("store has no index");
    let dim = engine.stats(&index_id).expect("stats").vector_dim;
    // The same deterministic query the keyboard spike's openprobe used.
    let query: Vec<f32> = (0..dim)
        .map(|i| ((i * 7919) % 200) as f32 / 100.0 - 1.0)
        .collect();
    let filter_value = json!({"category": {"$in": ["fact", "preference"]}});
    let filter_ref = filter.then_some(&filter_value);
    let plan = engine
        .prepare_query_text("the Lisbon trip window seat", &mode)
        .expect("prepare_query_text");
    let search = |engine: &CoreEngine| {
        engine
            .search_embedded_text(&index_id, &plan, Some(&query), top_k, filter_ref)
            .expect("search")
    };

    let probe = Probe::start();
    let first = search(&engine);
    probe.report("search1");

    let probe = Probe::start();
    let second = search(&engine);
    probe.report("search2");
    assert_eq!(first, second, "a warm search must repeat the first one");

    let mut digest: u64 = 0xcbf2_9ce4_8422_2325;
    for hit in &first.hits {
        for byte in hit
            .document_id
            .bytes()
            .chain(hit.chunk_id.bytes())
            .chain(hit.score.to_bits().to_le_bytes())
        {
            digest = (digest ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    println!(
        "hits {} considered {} digest {digest:016x}",
        first.hits.len(),
        first.total_considered
    );
    if hold_secs > 0 {
        // Keep the handle open so vmmap / footprint can inspect the process.
        println!("pid {}", std::process::id());
        std::thread::sleep(std::time::Duration::from_secs(hold_secs));
    }
    drop(engine);
}
