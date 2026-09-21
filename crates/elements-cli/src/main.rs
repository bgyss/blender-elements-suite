#![forbid(unsafe_code)]

//! Headless entry point for the Elements engine.
//!
//! Everything here runs without a window, without Blender, and without the
//! daemon, which is what lets CI exercise the engine on software Vulkan.

mod bake;
mod preview;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "elements", version, about = "Elements Suite headless engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Evaluate a graph and write one .vdb per frame.
    Bake {
        graph: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Frame or inclusive range (e.g. `5` or `1-100`). Output filenames
        /// zero-pad the frame number to AT LEAST four digits and grow beyond
        /// that past frame 9999 (e.g. `density.10000.vdb`), so consumers
        /// must parse the frame number rather than sort filenames as
        /// strings.
        #[arg(long, default_value = "1")]
        frames: String,
        #[arg(long, default_value = "density")]
        name: String,
        #[arg(long, default_value_t = 0.1)]
        voxel_size: f64,
    },
    /// Write one z-slice of a graph's output as a PNG.
    RenderPreview {
        graph: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        slice_z: Option<u32>,
        #[arg(long, default_value = "-1,1", allow_hyphen_values = true)]
        range: String,
    },
    /// Write a graph's output as a .npy array, for golden tests.
    DumpNpy {
        graph: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

fn main() {
    if let Err(e) = run() {
        // `{:#}` prints the whole anyhow context chain on one line.
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    match Cli::parse().command {
        Commands::Bake {
            graph,
            out,
            frames,
            name,
            voxel_size,
        } => bake::bake(
            &graph,
            &out,
            bake::parse_frames(&frames)?,
            &name,
            voxel_size,
        ),
        Commands::RenderPreview {
            graph,
            out,
            slice_z,
            range,
        } => preview::render_preview(&graph, &out, slice_z, preview::parse_range(&range)?),
        Commands::DumpNpy { graph, out } => preview::dump_npy(&graph, &out),
    }
}
