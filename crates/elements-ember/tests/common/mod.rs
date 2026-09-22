#![allow(dead_code)]

use elements_core::gpu::{Field, FieldDims, FieldFormat, FieldPool, GpuContext};

pub fn gpu() -> GpuContext {
    GpuContext::new_headless().expect("no GPU adapter available")
}

/// Index of voxel `(i, j, k)` in an x-fastest array of `dims`.
pub fn index(dims: FieldDims, i: u32, j: u32, k: u32) -> usize {
    (i + dims.x * (j + dims.y * k)) as usize
}

/// A pooled R32Float field holding `values`.
pub fn upload(gpu: &GpuContext, pool: &mut FieldPool, dims: FieldDims, values: &[f32]) -> Field {
    let field = pool.acquire(gpu, dims, FieldFormat::R32Float).unwrap();
    field.write(gpu, values).unwrap();
    field
}

/// A deterministic, irregular test pattern with values in about [-1, 1].
pub fn pattern(dims: FieldDims, seed: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity(dims.voxel_count());
    for k in 0..dims.z {
        for j in 0..dims.y {
            for i in 0..dims.x {
                let h = (i * 73 + j * 151 + k * 283 + seed * 997) % 211;
                out.push(h as f32 / 105.0 - 1.0);
            }
        }
    }
    out
}

pub fn assert_close(gpu: &[f32], cpu: &[f32], tol: f32, what: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{what}: length");
    for (n, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        assert!(
            (g - c).abs() <= tol,
            "{what}: element {n}: gpu {g}, cpu {c}"
        );
    }
}
