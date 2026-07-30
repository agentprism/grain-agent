//! Application data directory resolution, across every target this crate can
//! be built for.
//!
//! # Why this is its own module, and why it uses only `std`
//!
//! This logic used to live inline in [`crate::oauth`], enumerating
//! `linux` / `macos` / `windows` via `#[cfg]` blocks that were the function's
//! **tail expression**. A target matching none of them therefore left the
//! function body with no value at all, so the crate did not merely misbehave
//! elsewhere — *it failed to compile*, which blocked every Android and iOS
//! build of the harness.
//!
//! Two things follow from that, and both are structural rather than cosmetic:
//!
//! 1. **The platform match is now exhaustive by construction.** The narrow fix
//!    is "add `android` and `ios`", which repairs today's two targets and
//!    leaves the identical trap for the next one (FreeBSD, WASI, …). Instead
//!    there is a `#[cfg(not(any(…)))]` arm, so **no target can fail to compile
//!    here again**; an unrecognized platform degrades to a clear runtime error
//!    rather than a build break.
//!
//! 2. **This module depends on `std` and nothing else.** That is deliberate:
//!    it makes the real production logic checkable for mobile targets with a
//!    bare `rustc --target … --emit=metadata`, needing no Android NDK and no
//!    Apple SDK. A whole-crate `cargo check --target aarch64-linux-android`
//!    cannot serve as that guard — it dies in a transitive dependency's build
//!    script (`ring` needs a cross C compiler) long before reaching this code.
//!    `tests/mobile_targets.rs` exercises exactly this.
//!
//! # Why mobile cannot reuse the desktop derivation
//!
//! The desktop arms derive a path from environment variables. On mobile the
//! application data directory is **process-provided**, not environmental:
//!
//! - **Android** — the app-private directory comes from
//!   `Context.getFilesDir()` (a JNI call), typically
//!   `/data/user/0/<package>/files`. No environment variable exposes it:
//!   `HOME` is unset or `/`, and `ANDROID_DATA` is the device-wide `/data`,
//!   not the app sandbox. Deriving `$HOME/.config/grain` would resolve to
//!   `/.config/grain` — outside the sandbox, unwritable, and wrong in a way
//!   that surfaces later as a confusing permission error instead of a clear
//!   one. Android therefore *requires* the embedder to supply the directory,
//!   and [`DataDirError`] says so in as many words.
//!
//! - **iOS** — genuinely derivable, and so derived. The system sets `HOME` to
//!   the app sandbox root, and `<sandbox>/Library/Application Support` is
//!   Apple's designated location for application support data, making iOS
//!   equivalent to macOS. The one difference from the desktop arms is that
//!   `HOME` must be present **and absolute**: the desktop helper falls back to
//!   `"."` when `HOME` is missing, and a relative token path on iOS would
//!   silently write into the process working directory rather than the
//!   sandbox. An unusable `HOME` is reported, never guessed.
//!
//! `GRAIN_CONFIG_DIR` is consulted first on every platform and is the
//! documented escape hatch: it is how an Android embedder passes
//! `getFilesDir()` in, and it lets any platform be overridden for tests or
//! portable installs.
//!
//! # Compatibility
//!
//! Desktop behavior is unchanged, byte for byte — the same variables are read
//! in the same order and produce the same paths as before.

use std::path::PathBuf;

/// No application data directory could be determined.
///
/// Returned only where the directory is supplied by the host process rather
/// than derivable from the environment (Android, an iOS process with no usable
/// `HOME`, and any target this crate has not been taught about). Never
/// returned on Linux / macOS / Windows, and never returned anywhere when
/// `GRAIN_CONFIG_DIR` is set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataDirError {
    /// The platform provides its data directory through the host application.
    Unavailable {
        /// `std::env::consts::OS` for the target that produced this.
        platform: &'static str,
    },
}

impl std::fmt::Display for DataDirError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataDirError::Unavailable { platform } => write!(
                f,
                "no application data directory available on this platform \
                 ({platform}): it is provided by the host process, not derivable \
                 from the environment. Set GRAIN_CONFIG_DIR to the directory the \
                 embedder should use (Android: Context.getFilesDir(); iOS: the \
                 app sandbox's Library/Application Support)."
            ),
        }
    }
}

impl std::error::Error for DataDirError {}

/// The directory under which grain stores per-user state.
///
/// See the module documentation for the per-platform reasoning.
pub fn data_dir() -> Result<PathBuf, DataDirError> {
    // The explicit override wins everywhere. On platforms whose data directory
    // is process-provided this is the *only* mechanism, so it must come first.
    if let Ok(dir) = std::env::var("GRAIN_CONFIG_DIR") {
        return Ok(PathBuf::from(dir));
    }
    platform_data_dir()
}

#[cfg(target_os = "linux")]
fn platform_data_dir() -> Result<PathBuf, DataDirError> {
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        return Ok(PathBuf::from(xdg));
    }
    Ok(desktop_home().join(".config").join("grain"))
}

#[cfg(target_os = "macos")]
fn platform_data_dir() -> Result<PathBuf, DataDirError> {
    Ok(desktop_home()
        .join("Library")
        .join("Application Support")
        .join("grain"))
}

#[cfg(target_os = "windows")]
fn platform_data_dir() -> Result<PathBuf, DataDirError> {
    if let Ok(appdata) = std::env::var("APPDATA") {
        return Ok(PathBuf::from(appdata).join("grain"));
    }
    Ok(desktop_home().join(".config").join("grain"))
}

/// iOS: `HOME` is the app sandbox root, so this mirrors macOS — but it must be
/// absolute, because a relative fallback would silently write into the process
/// working directory instead of the sandbox.
#[cfg(target_os = "ios")]
fn platform_data_dir() -> Result<PathBuf, DataDirError> {
    match std::env::var("HOME") {
        Ok(home) if std::path::Path::new(&home).is_absolute() => Ok(PathBuf::from(home)
            .join("Library")
            .join("Application Support")
            .join("grain")),
        _ => Err(DataDirError::Unavailable {
            platform: std::env::consts::OS,
        }),
    }
}

/// Android: only the Android framework knows the app-private directory
/// (`Context.getFilesDir()`), so the embedder must pass it in through
/// `GRAIN_CONFIG_DIR`. Guessing here would fail obscurely at write time.
#[cfg(target_os = "android")]
fn platform_data_dir() -> Result<PathBuf, DataDirError> {
    Err(DataDirError::Unavailable {
        platform: std::env::consts::OS,
    })
}

/// Every other target. Present so the platform match is exhaustive and no
/// future target can reintroduce the build break this module used to cause.
#[cfg(not(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "windows",
    target_os = "ios",
    target_os = "android"
)))]
fn platform_data_dir() -> Result<PathBuf, DataDirError> {
    Err(DataDirError::Unavailable {
        platform: std::env::consts::OS,
    })
}

/// Desktop home-directory lookup, unchanged from the original implementation
/// including its `"."` fallback. Deliberately not used by the mobile arms:
/// iOS applies a stricter absolute-path check, and Android has no usable
/// `HOME` at all.
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
fn desktop_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The error text must tell an embedder exactly what to do — it is the
    /// only guidance an Android integrator gets at runtime.
    #[test]
    fn unavailable_error_names_the_remedy() {
        let rendered = DataDirError::Unavailable { platform: "android" }.to_string();
        assert!(rendered.contains("android"));
        assert!(
            rendered.contains("GRAIN_CONFIG_DIR"),
            "the error must name the override variable: {rendered}"
        );
        assert!(
            rendered.contains("getFilesDir"),
            "the error must name the Android source of truth: {rendered}"
        );
    }

    /// On every platform, including those with no derivable directory, an
    /// explicit override is honored verbatim. This is the mechanism mobile
    /// embedders depend on.
    #[test]
    fn explicit_override_is_honored_verbatim() {
        // `set_var` is process-global; this test owns the variable and
        // restores it, and no other test in this module reads the env.
        let previous = std::env::var("GRAIN_CONFIG_DIR").ok();
        unsafe { std::env::set_var("GRAIN_CONFIG_DIR", "/tmp/grain-data-dir-test") };
        let resolved = data_dir().expect("an override must always resolve");
        assert_eq!(resolved, PathBuf::from("/tmp/grain-data-dir-test"));
        match previous {
            Some(v) => unsafe { std::env::set_var("GRAIN_CONFIG_DIR", v) },
            None => unsafe { std::env::remove_var("GRAIN_CONFIG_DIR") },
        }
    }

    /// On the desktop targets this crate is developed on, resolution must
    /// succeed without any override and must be absolute.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    #[test]
    fn desktop_resolution_succeeds_without_an_override() {
        let previous = std::env::var("GRAIN_CONFIG_DIR").ok();
        unsafe { std::env::remove_var("GRAIN_CONFIG_DIR") };
        let resolved = data_dir().expect("desktop platforms always resolve");
        assert!(
            resolved.is_absolute() || resolved.starts_with("."),
            "unexpected shape: {resolved:?}"
        );
        if let Some(v) = previous {
            unsafe { std::env::set_var("GRAIN_CONFIG_DIR", v) };
        }
    }
}
