use std::path::PathBuf;
use std::process::Command;

#[test]
fn c_abi_smoke_compiles_and_runs() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir.join("../..");
    let target_dir = workspace.join("target/debug/deps");
    let source = manifest_dir.join("tests/abi_smoke.c");
    let temp_dir = std::env::temp_dir().join(format!("lodedb-ffi-smoke-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).expect("create smoke temp dir");
    let output = temp_dir.join("abi_smoke");
    let dylib_name = if cfg!(target_os = "macos") {
        "liblodedb_ffi.dylib"
    } else {
        "liblodedb_ffi.so"
    };
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
    let run = Command::new(&output).output().expect("run C smoke binary");
    assert!(
        run.status.success(),
        "C smoke failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let _ = std::fs::remove_dir_all(temp_dir);
}
