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

/// Every scene, resolution and solver the results table needs (spec §1).
pub const SCENES: [&str; 3] = ["plume", "plume_collider", "plume_wind"];
pub const RESOLUTIONS: [u32; 3] = [64, 128, 256];
pub const SOLVERS: [&str; 2] = ["ember", "mantaflow"];

/// A run whose load average was above this should be timed again idle.
pub const IDLE_LOAD: f64 = 2.0;

/// The frames the table reports, 1-based: the last emitting frame and the
/// last frame.
const REPORT_FRAMES: [usize; 2] = [60, 120];

/// Where and on what the results were produced, for the table's header.
#[derive(Debug, Clone, PartialEq)]
pub struct Context {
    pub machine: String,
    pub os: String,
    pub commit: String,
    pub blender: String,
    pub date: String,
}

/// The whole table, or the list of result files that are missing. Never a
/// partial table (spec §7).
///
/// Missing runs are named `{solver}-{scene}-{resolution}`, the stem of the
/// file [`summary_path`] would give.
pub fn results_markdown(summaries: &[RunSummary], ctx: &Context) -> Result<String, Vec<String>> {
    let find = |solver: &str, scene: &str, res: u32| {
        summaries
            .iter()
            .find(|s| s.solver == solver && s.scene == scene && s.resolution == res)
    };
    let mut missing = Vec::new();
    for solver in SOLVERS {
        for scene in SCENES {
            for res in RESOLUTIONS {
                if find(solver, scene, res).is_none() {
                    missing.push(format!("{solver}-{scene}-{res}"));
                }
            }
        }
    }
    if !missing.is_empty() {
        return Err(missing);
    }

    let mut md = String::new();
    let _ = write!(
        md,
        "# Ember against Mantaflow (2b-3)\n\n\
         - Machine: {}\n\
         - OS: {}\n\
         - Ember commit: {}\n\
         - Blender: {}\n\
         - Date: {}\n\n",
        ctx.machine, ctx.os, ctx.commit, ctx.blender, ctx.date
    );
    let loaded: Vec<String> = summaries_in_order(&find)
        .filter(|s| s.load_before > IDLE_LOAD || s.load_after > IDLE_LOAD)
        .map(|s| {
            format!(
                "`{}-{}-{}` ({:.1} → {:.1})",
                s.solver, s.scene, s.resolution, s.load_before, s.load_after
            )
        })
        .collect();
    let _ = writeln!(
        md,
        "Timings from a run whose 1-minute load average was above {IDLE_LOAD} before or \
         after its timed runs should be redone on an idle machine. Such runs: {}.\n",
        if loaded.is_empty() {
            "none".to_owned()
        } else {
            loaded.join(", ")
        }
    );
    md.push_str("## Summary\n\n_to be written by hand from the tables_\n\n");

    for scene in SCENES {
        let _ = write!(
            md,
            "## `{scene}`\n\n\
             | solver | cells | frame ms (median, min–max) | peak MiB \
             | div. RMS 1/s (60 / 120) | measured cells (60 / 120) \
             | kinetic energy m⁵/s² (60 / 120) | vorticity m³/s (60 / 120) \
             | centroid m (60 / 120) | top m (60 / 120) | drift at 120 (% of mass at 60) |\n\
             |---|---|---|---|---|---|---|---|---|---|---|\n"
        );
        for res in RESOLUTIONS {
            for solver in SOLVERS {
                let s = find(solver, scene, res).expect("checked above");
                md.push_str(&row(s));
            }
        }
        md.push('\n');
    }

    md.push_str(NOTES);
    Ok(md)
}

/// The summaries in table order: solver, then scene, then resolution.
fn summaries_in_order<'a>(
    find: &'a impl Fn(&str, &str, u32) -> Option<&'a RunSummary>,
) -> impl Iterator<Item = &'a RunSummary> + 'a {
    SOLVERS.into_iter().flat_map(move |solver| {
        SCENES.into_iter().flat_map(move |scene| {
            RESOLUTIONS
                .into_iter()
                .filter_map(move |res| find(solver, scene, res))
        })
    })
}

/// One table row.
fn row(s: &RunSummary) -> String {
    let at = |frame: usize| s.frames.get(frame - 1);
    let pair = |f: &dyn Fn(&FrameMetrics) -> Option<f64>| {
        let [a, b] = REPORT_FRAMES.map(|n| at(n).and_then(f).map_or("—".to_owned(), sig));
        format!("{a} / {b}")
    };
    let runs = if s.runs == 1 {
        "1 run".to_owned()
    } else {
        format!("median of {}", s.runs)
    };
    let cells = {
        let [a, b] =
            REPORT_FRAMES.map(|n| at(n).map_or("—".to_owned(), |f| f.measured_cells.to_string()));
        format!("{a} / {b}")
    };
    // drift[0] is frame 60, so frame 120 is drift[60].
    let drift = {
        let [first, last] = REPORT_FRAMES;
        match s.drift.get(last - first) {
            Some(&d) => {
                let base = at(first).map_or(0.0, |f| f.mass_below);
                if base > 0.0 {
                    format!("{} ({:+.1}%)", sig(d), 100.0 * d / base)
                } else {
                    format!("{} (—)", sig(d))
                }
            }
            None => "—".to_owned(),
        }
    };
    format!(
        "| {} | {}³ | {} ({}–{}), {runs} | {:.1} | {} | {cells} | {} | {} | {} | {} | {drift} |\n",
        s.solver,
        s.resolution,
        sig(s.frame_ms_median),
        sig(s.frame_ms_min),
        sig(s.frame_ms_max),
        s.peak_bytes as f64 / f64::from(1u32 << 20),
        pair(&|f| Some(f.divergence_rms)),
        pair(&|f| Some(f.kinetic_energy)),
        pair(&|f| Some(f.vorticity)),
        pair(&|f| f.centroid_m),
        pair(&|f| f.top_m),
    )
}

/// Three significant figures, in scientific notation outside [1e-3, 1e5).
fn sig(v: f64) -> String {
    let a = v.abs();
    if v == 0.0 || !v.is_finite() {
        return format!("{v}");
    }
    if (1e-3..1e5).contains(&a) {
        let decimals = (2 - a.log10().floor() as i32).max(0) as usize;
        format!("{v:.decimals$}")
    } else {
        format!("{v:.2e}")
    }
}

/// What every number in the table does and does not compare
/// (`docs/bench/mantaflow-notes.md`).
const NOTES: &str = "## Notes\n\n\
- **Velocity metrics** (divergence, kinetic energy, vorticity) cover only measured cells: \
those whose whole 3×3×3 neighbourhood is inside the domain, outside any collider and holds \
smoke (density > 1e-6). The rule is the same for both solvers, because Mantaflow's cache \
stores velocity only where there is smoke (spec §4.1). The measured-cell count says how \
much of each field that is.\n\
- **Wind.** Mantaflow's wind field acts only on cells that hold smoke; Ember's wind \
accelerates every cell. `plume_wind` compares the plume's shape, not a matched force field.\n\
- **Heat.** Mantaflow's emitter heat is held at a set value (Ember's emitter-centre heat at \
frame 24), not added at Ember's rate, so Mantaflow's emitter is hotter before frame 24 and \
cooler after it. Plume centroid and top carry that difference. Emitted mass matches within 3%.\n\
- **Pressure.** Mantaflow solves with multigrid-preconditioned conjugate gradients to a \
tolerance; Ember runs a fixed count of red-black Gauss–Seidel iterations.\n\
- **Drift** is (mass below the outflow plane + outflow since frame 60) − that mass at frame \
60, with outflow estimated at frame resolution as the net upwind flux through the z-faces two \
cells below the top, and mass summed over the layers below them (spec §4.3). Emission stops \
after frame 60, so a perfect solver drifts 0.\n\
- **Frame times** exclude frame 1. Ember's frame is `eval_frame` plus a blocking GPU wait; \
Mantaflow's is the difference between consecutive cache files' modification times, so it \
includes writing the cache. 256³ is one run of each solver, the other resolutions the median \
of three runs' medians; the min–max range pools every timed frame.\n\
- **Peak memory.** Ember's is the field pool's allocated bytes, with the frame cache off. \
Mantaflow's is Blender's peak resident memory while baking, minus the same scene's peak \
without a bake.\n";
