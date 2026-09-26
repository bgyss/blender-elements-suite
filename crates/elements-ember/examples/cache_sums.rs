//! Per-frame sums of named float grids in a Mantaflow cache, for the fire
//! probes in `docs/bench/mantaflow-notes.md` ("Fire").
//!
//! Run: cargo run -p elements-ember --example cache_sums -- CACHE_DIR RES FRAMES GRID...
//! Prints one line per frame: `frame grid=sum/max ...`, with "absent" for a
//! grid the frame does not hold.

// This example uses only the cache reader, not the timing helpers.
#[allow(dead_code)]
mod common;

use std::error::Error;
use std::path::PathBuf;

use common::mantaflow::read_float_grid;
use elements_core::gpu::FieldDims;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [dir, res, frames, grids @ ..] = args.as_slice() else {
        return Err("usage: cache_sums CACHE_DIR RES FRAMES GRID...".into());
    };
    let n: u32 = res.parse()?;
    let cells = FieldDims::new(n, n, n);
    for frame in 1..=frames.parse::<u32>()? {
        let path = PathBuf::from(dir).join(format!("data/fluid_data_{frame:04}.vdb"));
        let mut line = format!("{frame}");
        for grid in grids {
            match read_float_grid(&path, cells, grid)? {
                None => line.push_str(&format!(" {grid}=absent")),
                Some(v) => {
                    let sum: f64 = v.iter().map(|&x| f64::from(x)).sum();
                    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    line.push_str(&format!(" {grid}={sum:.6}/{max:.6}"));
                }
            }
        }
        println!("{line}");
    }
    Ok(())
}
