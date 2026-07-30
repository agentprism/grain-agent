//! Compile-check `src/data_dir.rs` for the mobile targets, without an SDK.
//!
//! # What this guards
//!
//! `data_dir.rs` used to be a `#[cfg]` cascade covering `linux` / `macos` /
//! `windows` only, whose blocks were the function's tail expression. On any
//! other target the function had no value and the **crate failed to compile**,
//! blocking every Android and iOS build of the harness. Nothing in the test
//! suite could see that, because tests only ever ran on the host.
//!
//! # Why it checks a module rather than the crate
//!
//! `cargo check -p grain-llm-genai --target aarch64-linux-android` cannot serve
//! as this guard: it dies in a transitive dependency's build script long before
//! reaching our code (`ring`, via rustls, invokes a cross C compiler —
//! `error occurred in cc-rs: failed to find tool "aarch64-linux-android-clang"`).
//! Making that work needs the Android NDK, which is not available here and is a
//! heavy CI dependency.
//!
//! `src/data_dir.rs` is therefore deliberately **std-only**, which lets a bare
//! `rustc --target … --emit=metadata` type-check the real production source for
//! each mobile target with no NDK and no Apple SDK. That is a narrower guard
//! than a full cross build — it proves this module compiles and its platform
//! match is exhaustive, not that the whole crate links — but it is precisely
//! the defect that occurred, and it costs nothing.
//!
//! # Skipping
//!
//! Targets whose `rust-std` is not installed are skipped with a printed note
//! rather than failing, so the suite stays green on a machine without them.
//! Install with e.g. `rustup target add aarch64-linux-android`. The test fails
//! if a target *is* installed and the module does not compile for it, and also
//! if **no** target could be checked at all — otherwise it could pass
//! vacuously.

use std::path::PathBuf;
use std::process::Command;

/// Targets that previously failed to compile, plus one non-mobile target to
/// prove the `#[cfg(not(any(…)))]` catch-all really is reached.
const TARGETS: &[&str] = &[
    "aarch64-linux-android",
    "armv7-linux-androideabi",
    "x86_64-linux-android",
    "aarch64-apple-ios",
    // Not mobile, and intentionally not enumerated in `data_dir.rs`: this one
    // only compiles because the catch-all arm exists.
    "wasm32-unknown-unknown",
];

fn module_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/data_dir.rs")
}

fn target_installed(target: &str) -> bool {
    Command::new("rustc")
        .args(["--print", "target-libdir", "--target", target])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| {
            let dir = String::from_utf8_lossy(&o.stdout).trim().to_string();
            !dir.is_empty() && PathBuf::from(dir).exists()
        })
        .unwrap_or(false)
}

/// Type-check the module for `target`; returns rustc's stderr on failure.
fn check(target: &str, out_dir: &std::path::Path) -> Result<(), String> {
    let out = out_dir.join(format!("{target}.rmeta"));
    let result = Command::new("rustc")
        .args([
            "--edition",
            "2024",
            "--crate-type",
            "lib",
            "--crate-name",
            "grain_data_dir_probe",
            "--target",
            target,
            "--emit=metadata",
            "-o",
        ])
        .arg(&out)
        .arg(module_path())
        .output()
        .map_err(|e| format!("failed to run rustc: {e}"))?;

    if result.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&result.stderr).to_string())
    }
}

#[test]
fn data_dir_module_compiles_for_mobile_and_unknown_targets() {
    let out_dir = std::env::temp_dir().join("grain-data-dir-target-check");
    std::fs::create_dir_all(&out_dir).expect("create probe output dir");

    let mut checked = Vec::new();
    let mut skipped = Vec::new();

    for target in TARGETS {
        if !target_installed(target) {
            skipped.push(*target);
            continue;
        }
        match check(target, &out_dir) {
            Ok(()) => checked.push(*target),
            Err(stderr) => panic!(
                "src/data_dir.rs does not compile for {target}.\n\
                 This is the defect that blocked every mobile build of the \
                 harness: a target with no matching `#[cfg]` arm leaves the \
                 function with no value.\n\nrustc said:\n{stderr}"
            ),
        }
    }

    if !skipped.is_empty() {
        println!(
            "skipped (rust-std not installed; `rustup target add <t>`): {}",
            skipped.join(", ")
        );
    }

    assert!(
        !checked.is_empty(),
        "no target could be checked, so this test proved nothing. Install at \
         least one of: {}",
        TARGETS.join(", ")
    );
    println!("compile-checked src/data_dir.rs for: {}", checked.join(", "));
}
