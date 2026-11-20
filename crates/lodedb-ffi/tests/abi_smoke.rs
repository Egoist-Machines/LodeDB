use std::path::{Path, PathBuf};
use std::process::Command;

#[test]
fn c_abi_smoke_compiles_and_runs() {
    run_c_abi_smoke(false);
}

#[test]
fn c_abi_smoke_compiles_and_runs_with_sanitizers() {
    if std::env::var("LODEDB_FFI_SANITIZERS").as_deref() != Ok("1") {
        eprintln!("set LODEDB_FFI_SANITIZERS=1 to run the C ABI sanitizer smoke test");
        return;
    }
    run_c_abi_smoke(true);
}

fn run_c_abi_smoke(sanitized: bool) {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir.join("../..");
    let source = manifest_dir.join("tests/abi_smoke.c");
    let suffix = if sanitized { "sanitized" } else { "plain" };
    let temp_dir =
        std::env::temp_dir().join(format!("lodedb-ffi-smoke-{}-{suffix}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).expect("create smoke temp dir");
    let output = temp_dir.join("abi_smoke");
    let dylib_name = if cfg!(target_os = "macos") {
        "liblodedb_ffi.dylib"
    } else {
        "liblodedb_ffi.so"
    };
    let target_dir = ffi_dylib_dir(&workspace.join("target/debug"), dylib_name);
    std::fs::copy(target_dir.join(dylib_name), temp_dir.join(dylib_name))
        .expect("copy ffi dylib beside smoke binary");
    let mut command = Command::new("cc");
    command
        .arg(&source)
        .arg("-I")
        .arg(manifest_dir.join("include"))
        .arg("-L")
        .arg(&target_dir)
        .arg("-llodedb_ffi")
        .arg("-o")
        .arg(&output);
    if sanitized {
        command.arg(if cfg!(target_os = "macos") {
            "-fsanitize=address"
        } else {
            "-fsanitize=address,leak"
        });
        command.arg("-fno-omit-frame-pointer");
    }
    if cfg!(target_os = "macos") {
        command.arg("-Wl,-rpath,@loader_path");
        command.arg("-framework").arg("Accelerate");
    } else {
        command.arg("-Wl,-rpath,$ORIGIN");
        command.arg("-lopenblas");
    }
    let compile = command.output().expect("run cc");
    assert!(
        compile.status.success(),
        "cc failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&compile.stdout),
        String::from_utf8_lossy(&compile.stderr)
    );
    let mut run_command = Command::new(&output);
    if sanitized {
        if cfg!(target_os = "macos") {
            run_command.env("ASAN_OPTIONS", "abort_on_error=1:strict_string_checks=1");
        } else {
            run_command.env(
                "ASAN_OPTIONS",
                "abort_on_error=1:detect_leaks=1:strict_string_checks=1",
            );
            run_command.env("LSAN_OPTIONS", "exitcode=23");
        }
    }
    let run = run_command.output().expect("run C smoke binary");
    assert!(
        run.status.success(),
        "C smoke failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let _ = std::fs::remove_dir_all(temp_dir);
}

/// The directory holding the newest build of the ffi dylib. Cargo through
/// 1.99 wrote it to `target/debug/deps`; from 1.100 each unit builds under
/// `target/debug/build/<package>/<hash>/out`. Both are searched, plus the
/// uplifted `target/debug` copy a `cargo build` leaves.
fn ffi_dylib_dir(debug_dir: &Path, dylib_name: &str) -> PathBuf {
    let mut candidates = vec![debug_dir.to_path_buf(), debug_dir.join("deps")];
    if let Ok(units) = std::fs::read_dir(debug_dir.join("build/lodedb-ffi")) {
        candidates.extend(units.flatten().map(|unit| unit.path().join("out")));
    }
    candidates
        .into_iter()
        .filter_map(|dir| {
            let modified = dir.join(dylib_name).metadata().ok()?.modified().ok()?;
            Some((modified, dir))
        })
        .max_by_key(|(modified, _)| *modified)
        .map(|(_, dir)| dir)
        .unwrap_or_else(|| panic!("no {dylib_name} under {}", debug_dir.display()))
}
