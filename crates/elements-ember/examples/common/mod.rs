//! Helpers shared by the bench examples. Cargo treats only `examples/*.rs`
//! and `examples/*/main.rs` as examples, so this is a module, not one.

use std::process::Command;

// Only the benchmark example reads Mantaflow caches.
#[allow(dead_code)]
pub mod mantaflow;

pub fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

pub fn shell(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".to_owned())
}

/// Like `shell`, but distinguishes "the command failed" from "it printed
/// nothing", so callers can tell a real empty result from a failure.
pub fn shell_checked(program: &str, args: &[&str]) -> Option<String> {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
}

/// The short commit hash, with `-dirty` when the tree has changes, for a
/// results table's header.
// speed_gate and presets use this; the benchmark uses `commit_label_excluding`.
#[allow(dead_code)]
pub fn commit_label() -> String {
    commit_label_excluding(&[])
}

/// Like `commit_label`, but changes under `excluded` (paths from the
/// repository root) do not make the tree dirty. A tool that writes into the
/// tree passes its own output directories, so its first run cannot mark
/// every later run dirty. Every other change, untracked files included,
/// still counts.
// Only the benchmark writes into the tree.
#[allow(dead_code)]
pub fn commit_label_excluding(excluded: &[&str]) -> String {
    let mut args = vec!["status".to_owned(), "--porcelain".to_owned()];
    if !excluded.is_empty() {
        args.extend(["--".to_owned(), ":/".to_owned()]);
        args.extend(excluded.iter().map(|p| format!(":(top,exclude){p}")));
    }
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    match shell_checked("git", &["rev-parse", "--short", "HEAD"]) {
        Some(hash) => match shell_checked("git", &args) {
            Some(status) if !status.is_empty() => format!("{hash}-dirty"),
            Some(_) => hash,
            // `git status` failed to run: don't claim a clean tree we didn't verify.
            None => format!("{hash} (dirty status unknown)"),
        },
        None => "unknown".to_owned(),
    }
}
