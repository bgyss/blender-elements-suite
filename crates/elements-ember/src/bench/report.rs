//! The Mantaflow benchmark's result files (2b-3 spec §6).
//!
//! Each run of either solver writes one `RunSummary` as JSON, with a CSV of
//! its per-frame metrics beside it for plotting. The report mode reads the
//! JSON files back to build the results table.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::metrics::FrameMetrics;

/// One solver's run of one scene at one resolution.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RunSummary {
    /// "ember" or "mantaflow".
    pub solver: String,
    pub scene: String,
    pub resolution: u32,
    /// Timed runs; the metrics come from one further, untimed run.
    pub runs: u32,
    /// Median over runs of each run's median frame time, frame 1 excluded.
    pub frame_ms_median: f64,
    pub frame_ms_min: f64,
    pub frame_ms_max: f64,
    /// Peak GPU (Ember) or process (Mantaflow) memory, bytes.
    pub peak_bytes: u64,
    /// 1-minute load average before and after the timed runs.
    pub load_before: f64,
    pub load_after: f64,
    /// Blender's version string for Mantaflow runs; `None` for Ember.
    pub blender: Option<String>,
    /// One entry per frame, frame 1 first.
    pub frames: Vec<FrameMetrics>,
    /// `drift` from frame 60, one entry per frame from 60 on.
    pub drift: Vec<f64>,
}

/// `{dir}/{solver}-{scene}-{resolution}.json`.
pub fn summary_path(dir: &Path, solver: &str, scene: &str, resolution: u32) -> PathBuf {
    dir.join(format!("{solver}-{scene}-{resolution}.json"))
}

/// Write `s` as JSON, and its per-frame metrics as a CSV with the same stem.
pub fn write_summary(dir: &Path, s: &RunSummary) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = summary_path(dir, &s.solver, &s.scene, s.resolution);
    let json = serde_json::to_string_pretty(s).map_err(std::io::Error::other)?;
    std::fs::write(&path, json + "\n")?;
    std::fs::write(path.with_extension("csv"), frames_csv(&s.frames))
}

/// A header row, then one row per frame (frame 1 first): `frame`, then each
/// `FrameMetrics` field in declaration order, with `None` as an empty cell.
fn frames_csv(frames: &[FrameMetrics]) -> String {
    let opt = |v: Option<f64>| v.map(|v| v.to_string()).unwrap_or_default();
    let mut out = String::from(
        "frame,divergence_max,divergence_rms,kinetic_energy,vorticity,measured_cells,\
         mass,mass_below,centroid_m,top_m,outflow_rate\n",
    );
    for (n, f) in frames.iter().enumerate() {
        let _ = writeln!(
            out,
            "{},{},{},{},{},{},{},{},{},{},{}",
            n + 1,
            f.divergence_max,
            f.divergence_rms,
            f.kinetic_energy,
            f.vorticity,
            f.measured_cells,
            f.mass,
            f.mass_below,
            opt(f.centroid_m),
            opt(f.top_m),
            f.outflow_rate,
        );
    }
    out
}

/// The 1-minute load average, from `sysctl -n vm.loadavg` (macOS prints
/// `{ 1.23 1.45 1.67 }`). Negative when it cannot be read, since JSON has
/// no NaN.
pub fn load_average() -> f64 {
    Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .split_whitespace()
                .find_map(|t| t.parse::<f64>().ok())
        })
        .unwrap_or(-1.0)
}
