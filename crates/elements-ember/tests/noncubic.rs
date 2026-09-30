mod common;

use common::*;

/// A 1 m × 1 m × 2 m domain: `dims` is [nx, ny, nz] and `domain_size` is the
/// longest axis, so dx = 2 / 64 = 1/32 m on every axis.
const DOC: &str = r#"{
  "version": 3,
  "dims": [32, 32, 64],
  "fps": 24.0,
  "start_frame": 1,
  "domain_size": 2.0,
  "nodes": [
    { "id": 0, "kind": "ember.sphere_emitter",
      "params": { "center": [0.5, 0.5, 0.3], "radius": 0.15, "density_rate": 1.0, "temperature_rate": 1.0 } },
    { "id": 1, "kind": "ember.smoke_solver",
      "params": { "substeps": 1, "pressure_iterations": 160, "buoyancy_density": 0.0, "buoyancy_temperature": 1.0 } },
    { "id": 2, "kind": "core.output", "params": {} }
  ],
  "edges": [
    { "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 },
    { "from_node": 0, "from_index": 1, "to_node": 1, "to_index": 1 },
    { "from_node": 1, "from_index": 0, "to_node": 2, "to_index": 0 }
  ],
  "output": 2
}"#;

/// Density-weighted centroid in metres, and the total, from x-fastest bits.
fn centroid(bits: &[u32]) -> ([f64; 3], f64) {
    let (nx, ny) = (32usize, 32usize);
    let dx = 1.0 / 32.0;
    let (mut c, mut total) = ([0.0f64; 3], 0.0f64);
    for (n, &b) in bits.iter().enumerate() {
        let d = f64::from(f32::from_bits(b));
        let (i, j, k) = (n % nx, (n / nx) % ny, n / (nx * ny));
        c[0] += d * (i as f64 + 0.5) * dx;
        c[1] += d * (j as f64 + 0.5) * dx;
        c[2] += d * (k as f64 + 0.5) * dx;
        total += d;
    }
    (c.map(|v| v / total), total)
}

/// Every axis gets the same dx, so a plume centred in x and y rises along z
/// and stays centred. A solver that mixed up axis lengths would drift it.
#[test]
fn a_tall_domain_runs_and_the_plume_rises_along_its_long_axis() {
    let mut s = Session::new(DOC);
    let mut tl = timeline(0);
    let early = s.density_bits(&mut tl, 10);
    let late = s.density_bits(&mut tl, 40);
    assert_eq!(late.len(), 32 * 32 * 64, "output has the document's dims");
    assert!(
        late.iter().all(|&b| f32::from_bits(b).is_finite()),
        "density must stay finite"
    );
    let ([_, _, z_early], _) = centroid(&early);
    let ([x, y, z_late], total) = centroid(&late);
    assert!(total > 0.0, "the plume must exist");
    assert!(z_late > z_early + 0.1, "rises: {z_early} -> {z_late}");
    let tol = 1.0 / 32.0;
    assert!((x - 0.5).abs() < tol, "x centroid {x}");
    assert!((y - 0.5).abs() < tol, "y centroid {y}");
}
