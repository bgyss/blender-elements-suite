//! Helpers shared by the bench examples. Cargo treats only `examples/*.rs`
//! and `examples/*/main.rs` as examples, so this is a module, not one.

use std::process::Command;

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
pub fn commit_label() -> String {
    match shell_checked("git", &["rev-parse", "--short", "HEAD"]) {
        Some(hash) => match shell_checked("git", &["status", "--porcelain"]) {
            Some(status) if !status.is_empty() => format!("{hash}-dirty"),
            Some(_) => hash,
            // `git status` failed to run: don't claim a clean tree we didn't verify.
            None => format!("{hash} (dirty status unknown)"),
        },
        None => "unknown".to_owned(),
    }
}
