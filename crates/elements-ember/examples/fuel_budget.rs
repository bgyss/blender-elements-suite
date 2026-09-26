//! Inspect a resumable Mantaflow fire cache's apparent fuel balance.
//!
//! Run: cargo run --release -p elements-ember --example fuel_budget -- CACHE_DIR RES FRAMES
//! `fuel_inflow` is an intermediate grid clipped to density on export. Its
//! difference from the preceding cached fuel is not necessarily emission;
//! the post-burn residual is not necessarily advection alone. Compare caches
//! with different clipping settings before interpreting either term.

#[allow(dead_code)]
mod common;

use std::error::Error;
use std::path::PathBuf;

use common::mantaflow::read_float_grid;
use elements_core::gpu::FieldDims;

const BURN_PER_FRAME: f64 = 0.078125;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [dir, res, frames] = args.as_slice() else {
        return Err("usage: fuel_budget CACHE_DIR RES FRAMES".into());
    };
    let n: u32 = res.parse()?;
    let cells = FieldDims::new(n, n, n);
    let mut previous = vec![0.0_f32; (n as usize).pow(3)];
    let mut inflow_delta = 0.0_f64;
    let mut burned = 0.0_f64;
    let mut residual = 0.0_f64;
    println!(
        "frame fuel inflow_delta visible_burn postburn_residual cumulative_inflow_delta cumulative_visible_burn cumulative_residual top_fuel"
    );
    for frame in 1..=frames.parse::<u32>()? {
        let path = PathBuf::from(dir).join(format!("data/fluid_data_{frame:04}.vdb"));
        let fuel = read_float_grid(&path, cells, "fuel")?.ok_or("missing fuel grid")?;
        let inflow =
            read_float_grid(&path, cells, "fuel_inflow")?.ok_or("missing fuel_inflow grid")?;
        if frame == 1 {
            let mut min = [n; 3];
            let mut max = [0; 3];
            let mut count = 0;
            for (idx, &v) in inflow.iter().enumerate() {
                if v <= 0.0 {
                    continue;
                }
                let p = [idx as u32 % n, (idx as u32 / n) % n, idx as u32 / (n * n)];
                for axis in 0..3 {
                    min[axis] = min[axis].min(p[axis]);
                    max[axis] = max[axis].max(p[axis]);
                }
                count += 1;
            }
            eprintln!("frame 1 inflow: {count} cells, bounds {min:?}..={max:?}");
        }
        let e: f64 = inflow
            .iter()
            .zip(&previous)
            .map(|(&input, &prior)| f64::from(input) - f64::from(prior))
            .sum();
        let b: f64 = inflow
            .iter()
            .map(|&x| f64::from(x).clamp(0.0, BURN_PER_FRAME))
            .sum();
        let input_sum: f64 = inflow.iter().map(|&x| f64::from(x)).sum();
        let fuel_sum: f64 = fuel.iter().map(|&x| f64::from(x)).sum();
        let a = fuel_sum - input_sum + b;
        let top_start = ((n - 1) as usize) * (n as usize).pow(2);
        let top: f64 = fuel[top_start..].iter().map(|&x| f64::from(x)).sum();
        inflow_delta += e;
        burned += b;
        residual += a;
        println!(
            "{frame} {fuel_sum:.3} {e:.3} {b:.3} {a:.3} {inflow_delta:.3} {burned:.3} {residual:.3} {top:.3}"
        );
        previous = fuel;
    }
    Ok(())
}
