# Flamethrower vs. shack, FT0–FT3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundations for the flamethrower-vs-shack shot on Ember: verify non-cubic domains and fix the shot's scene (FT0), add a nozzle jet emitter (FT1), add a mesh collider that handles concave, thin-walled geometry (FT2), and measure the fire quality gap (FT3).

**Architecture:** `Shape` (shared by emitters and colliders) gains a `Cone` variant and the emitter gains a local-frame velocity option. A new `ember.mesh_collider` node turns an inline triangle mesh into the same `(sdf, face velocity)` pair `ember.collider` produces, using a brute-force GPU closest-triangle kernel, so unions and the solver need no change. A procedural shack mesh generator supplies the shot's collider. FT3 is an experiment track that ends in a recorded decision, not a solver change.

**Tech Stack:** Rust (wgpu 30, WGSL), `elements-ember` crate, cargo-nextest, `just check`.

**Spec:** `docs/superpowers/specs/2026-09-30-flamethrower-napalm-roadmap-design.md` (§4, FT0–FT3). FT4 (surface ignition), FT5 (multi-grid export) and FT6 (shot assembly) get their own plans after this one lands.

## Global Constraints

- `crates/*` are `Apache-2.0 OR MIT`; **no GPL code may be copied into any `elements-*` crate.** The closest-point-on-triangle routine is Ericson's published algorithm (*Real-Time Collision Detection*), reimplemented here, not vendored.
- `#![forbid(unsafe_code)]` stays in `elements-ember`.
- `required_features` stays `wgpu::Features::empty()`; scalar fields are `R32Float`; at most 4 storage textures per shader stage (storage *buffers* are fine; read fields through `texture_3d<f32>` + `textureLoad`).
- Crate manifests use `dep.workspace = true`; add no new dependencies.
- Kernels with solids include `solid.wgsl` and declare `var solid: texture_3d<f32>` (not touched here).
- Every stochastic node takes an explicit `seed: u64`. No implicit entropy (the shack's breakage takes a seed).
- Fire-off behaviour stays bit-identical: nothing in this plan may change fire-off output. `just check` includes the fire-off hash guard on the M1 Max.
- **Never set `WGPU_BACKEND` locally** (macOS/Metal). CI runs on llvmpipe; check `gh run list` after pushing.
- Tests: **prove each test can fail.** Mutate the code it covers, watch it fail, restore, and record the real output. A mutation changes exactly one thing.
- Verify wgpu APIs in the vendored crate source, not docs.rs (`~/.cargo/registry/src/*/wgpu-types-30.0.1`).
- Commits: plain imperative subject, body explains why, ends with the attribution lines the harness specifies. Plain `git`, not `jj`. `just check` passes before every commit.
- Code comments match the surrounding density: doc comments on public items, a one-line reference to the spec where it decides behaviour.

## Review Focus

1. **Non-cubic domain.** `dims [32,32,64]` (and the shot's `[256,128,128]`) must run; a failure here is a finding, not a skipped test (Task 1).
2. **Bad meshes.** Empty mesh, index out of range, NaN/inf position, length not divisible by 3, and a mesh over the triangle cap must each be rejected with a named `DocError`, never panic or hang the GPU (Tasks 2 and 6).
3. **Thin planks.** A plank thinner than one voxel must still block flow when `offset` is set, and the test must fail without it (Task 7).
4. **Degenerate and inverted triangles.** A zero-area triangle in an otherwise closed mesh must not turn the SDF into NaN; it is skipped (Task 6).
5. **Rotated nozzle.** A cone rotated 90° about z with `velocity_local: true` must push along +y, and a world-frame velocity must ignore the rotation (Task 5); an off-frame jet's silence is already covered by `tests/shape_emitter.rs`.
6. **Cone with a zero radius at one end** (a true cone) must give an SDF equal to the distance to the tip (Task 4).

---

### Task 1: Non-cubic domains run (FT0)

**Files:**
- Create: `crates/elements-ember/tests/noncubic.rs`

**Interfaces:**
- Consumes: `common::{Session, timeline}` (`tests/common/mod.rs`): `Session::new(doc: &str)`, `Session::density_bits(&mut self, &mut Timeline, frame: u32) -> Vec<u32>`, `timeline(budget_bytes: u64) -> Timeline`; `FieldDims` fields `x, y, z`.
- Produces: a passing characterization test, or a finding (see Step 3).

- [ ] **Step 1: Write the test**

```rust
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
```

- [ ] **Step 2: Run it**

Run: `cargo nextest run -p elements-ember --test noncubic`
Expected: PASS. If the document fails to load or the solver returns an error (for example multigrid needing equal axes), that is the finding: **stop**, record the exact error in `.superpowers/sdd/progress.md` and in the FT0 shot spec (Task 3), and ask the user whether to fix the solver or fall back to a cubic domain. Do not skip the test.

- [ ] **Step 3: Prove it can fail (single mutation, in the test)**

Change the emitter `center` x from `0.5` to `0.25` in `DOC`. Run the test; expected FAIL at `x centroid` (about 0.26 vs 0.5). Restore. Record the output in the commit body.

- [ ] **Step 4: Commit**

```bash
just check
git add crates/elements-ember/tests/noncubic.rs
git commit -m "Test that a non-cubic domain runs"
```

---

### Task 2: A triangle mesh type (FT0/FT2 foundation)

**Files:**
- Create: `crates/elements-ember/src/mesh.rs`
- Modify: `crates/elements-ember/src/lib.rs` (add `pub mod mesh;`)
- Test: `crates/elements-ember/tests/mesh.rs`

**Interfaces:**
- Produces (used by Tasks 3, 6–8):
  - `pub struct Mesh { pub positions: Vec<[f32; 3]>, pub indices: Vec<u32> }` — `Clone, Debug, PartialEq, Serialize, Deserialize`, `#[serde(deny_unknown_fields)]`.
  - `pub const MAX_TRIANGLES: usize = 1_000_000;`
  - `impl Mesh { pub fn validate(&self, kind: &str) -> Result<(), DocError>; pub fn triangle_count(&self) -> usize; pub fn box_mesh(min: [f32; 3], max: [f32; 3]) -> Mesh; pub fn merge(&mut self, other: &Mesh); pub fn from_obj(text: &str) -> Result<Mesh, String>; pub fn bounds(&self) -> ([f32; 3], [f32; 3]); }`
- Consumes: `crate::params::{bad, finite}` (`params::finite(kind, name, &[f32]) -> Result<(), DocError>`, `params::bad(kind, impl Into<String>) -> DocError`).

- [ ] **Step 1: Write the failing tests** (`tests/mesh.rs`)

```rust
use elements_ember::mesh::{MAX_TRIANGLES, Mesh};

fn outward_volume(m: &Mesh) -> f64 {
    // Signed volume by the divergence theorem: positive for outward winding.
    m.indices
        .chunks(3)
        .map(|t| {
            let [a, b, c] = [0, 1, 2].map(|i| m.positions[t[i] as usize].map(f64::from));
            (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                / 6.0
        })
        .sum()
}

#[test]
fn a_box_mesh_is_closed_and_wound_outward() {
    let m = Mesh::box_mesh([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]);
    assert_eq!(m.triangle_count(), 12);
    assert!(
        (outward_volume(&m) - 6.0).abs() < 1e-9,
        "volume {}",
        outward_volume(&m)
    );
    m.validate("test").unwrap();
}

#[test]
fn merging_offsets_the_second_meshs_indices() {
    let mut a = Mesh::box_mesh([0.0; 3], [1.0; 3]);
    let b = Mesh::box_mesh([2.0; 3], [3.0; 3]);
    a.merge(&b);
    assert_eq!(a.positions.len(), 16);
    assert_eq!(a.triangle_count(), 24);
    assert!(a.indices[36..].iter().all(|&i| i >= 8));
    assert!((outward_volume(&a) - 2.0).abs() < 1e-9);
}

#[test]
fn bad_meshes_are_rejected_with_a_reason() {
    let ok = Mesh::box_mesh([0.0; 3], [1.0; 3]);
    let cases: Vec<(&str, Mesh)> = vec![
        ("no triangles", Mesh { positions: vec![], indices: vec![] }),
        ("not a multiple of 3", Mesh { indices: vec![0, 1], ..ok.clone() }),
        ("out of range", Mesh { indices: vec![0, 1, 99], ..ok.clone() }),
        (
            "non-finite",
            Mesh {
                positions: {
                    let mut p = ok.positions.clone();
                    p[3][1] = f32::NAN;
                    p
                },
                ..ok.clone()
            },
        ),
        (
            "too many triangles",
            Mesh {
                positions: ok.positions.clone(),
                indices: vec![0; (MAX_TRIANGLES + 1) * 3],
            },
        ),
    ];
    for (what, m) in cases {
        let e = m.validate("ember.mesh_collider").unwrap_err().to_string();
        assert!(e.contains("ember.mesh_collider"), "{what}: {e}");
    }
}

#[test]
fn obj_text_loads_with_slashed_indices_and_polygons() {
    let obj = "# a quad and a triangle\n\
               v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0 0 1\n\
               vt 0 0\nvn 0 0 1\n\
               f 1/1/1 2/1/1 3/1/1 4/1/1\n\
               f 1 2 5\n";
    let m = Mesh::from_obj(obj).unwrap();
    assert_eq!(m.positions.len(), 5);
    assert_eq!(m.triangle_count(), 3, "the quad fans into two triangles");
    assert_eq!(&m.indices[..6], &[0, 1, 2, 0, 2, 3]);
    assert!(Mesh::from_obj("f 1 2 3\n").is_err(), "indices need vertices");
    assert!(Mesh::from_obj("v 0 0 0\nf -1 -1 -1\n").is_err(), "negative indices unsupported");
}

#[test]
fn bounds_are_the_axis_aligned_extent() {
    let m = Mesh::box_mesh([-1.0, 0.5, 2.0], [3.0, 1.5, 4.0]);
    assert_eq!(m.bounds(), ([-1.0, 0.5, 2.0], [3.0, 1.5, 4.0]));
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo nextest run -p elements-ember --test mesh`
Expected: FAIL to compile (`unresolved import elements_ember::mesh`).

- [ ] **Step 3: Implement `src/mesh.rs`**

```rust
//! Triangle meshes for `ember.mesh_collider` and the procedural shack
//! (flamethrower roadmap spec §4, FT2). Positions are in the object's local
//! space, metres; triangles wind counter-clockwise seen from outside.

use elements_core::graph::DocError;
use serde::{Deserialize, Serialize};

use crate::params;

/// The most triangles a mesh may hold. The SDF kernel tests every triangle at
/// every cell, so this bounds a frame's cost and the buffer size.
pub const MAX_TRIANGLES: usize = 1_000_000;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mesh {
    pub positions: Vec<[f32; 3]>,
    /// Three vertex indices per triangle.
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn validate(&self, kind: &str) -> Result<(), DocError> {
        if self.indices.is_empty() {
            return Err(params::bad(kind, "mesh has no triangles"));
        }
        if self.indices.len() % 3 != 0 {
            return Err(params::bad(
                kind,
                format!("mesh index count {} is not a multiple of 3", self.indices.len()),
            ));
        }
        if self.triangle_count() > MAX_TRIANGLES {
            return Err(params::bad(
                kind,
                format!(
                    "mesh has {} triangles, at most {MAX_TRIANGLES}",
                    self.triangle_count()
                ),
            ));
        }
        for p in &self.positions {
            params::finite(kind, "mesh position", p)?;
        }
        let n = self.positions.len();
        if let Some(&i) = self.indices.iter().find(|&&i| i as usize >= n) {
            return Err(params::bad(
                kind,
                format!("mesh index {i} is out of range for {n} positions"),
            ));
        }
        Ok(())
    }

    /// Axis-aligned bounds, `(min, max)`. Needs at least one position.
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for p in &self.positions {
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        (lo, hi)
    }

    /// A closed, outward-wound box: vertex `i` is `min` or `max` per axis by
    /// the bits of `i` (x = 1, y = 2, z = 4).
    pub fn box_mesh(min: [f32; 3], max: [f32; 3]) -> Mesh {
        let positions = (0..8u32)
            .map(|i| {
                [
                    if i & 1 == 0 { min[0] } else { max[0] },
                    if i & 2 == 0 { min[1] } else { max[1] },
                    if i & 4 == 0 { min[2] } else { max[2] },
                ]
            })
            .collect();
        let indices = vec![
            0, 4, 6, 0, 6, 2, // -x
            1, 3, 7, 1, 7, 5, // +x
            0, 1, 5, 0, 5, 4, // -y
            2, 6, 7, 2, 7, 3, // +y
            0, 2, 3, 0, 3, 1, // -z
            4, 5, 7, 4, 7, 6, // +z
        ];
        Mesh { positions, indices }
    }

    /// Append `other`, offsetting its indices.
    pub fn merge(&mut self, other: &Mesh) {
        let base = self.positions.len() as u32;
        self.positions.extend_from_slice(&other.positions);
        self.indices.extend(other.indices.iter().map(|&i| i + base));
    }

    /// Parse Wavefront OBJ text: `v` and `f` lines only. Faces may use
    /// `v/vt/vn` indices and any vertex count (fanned into triangles).
    /// Negative (relative) indices are not supported.
    pub fn from_obj(text: &str) -> Result<Mesh, String> {
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        for (n, line) in text.lines().enumerate() {
            let mut words = line.split_whitespace();
            match words.next() {
                Some("v") => {
                    let v: Vec<f32> = words.take(3).map(str::parse).collect::<Result<_, _>>()
                        .map_err(|e| format!("line {}: {e}", n + 1))?;
                    if v.len() != 3 {
                        return Err(format!("line {}: vertex needs 3 coordinates", n + 1));
                    }
                    positions.push([v[0], v[1], v[2]]);
                }
                Some("f") => {
                    let face: Vec<u32> = words
                        .map(|w| {
                            let i: i64 = w
                                .split('/')
                                .next()
                                .unwrap_or("")
                                .parse()
                                .map_err(|e| format!("line {}: {e}", n + 1))?;
                            if i < 1 || i as usize > positions.len() {
                                return Err(format!(
                                    "line {}: index {i} needs an earlier positive vertex",
                                    n + 1
                                ));
                            }
                            Ok((i - 1) as u32)
                        })
                        .collect::<Result<_, String>>()?;
                    if face.len() < 3 {
                        return Err(format!("line {}: face needs 3 vertices", n + 1));
                    }
                    for k in 1..face.len() - 1 {
                        indices.extend_from_slice(&[face[0], face[k], face[k + 1]]);
                    }
                }
                _ => {}
            }
        }
        Ok(Mesh { positions, indices })
    }
}
```

Add `pub mod mesh;` to `lib.rs` in alphabetical position (after `pub mod kernels;`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo nextest run -p elements-ember --test mesh`
Expected: 5 passed. (If `DocError`'s `Display` does not contain the kind string, adjust the assertion to whatever `params::bad` produces; check `src/params.rs`.)

- [ ] **Step 5: Prove the tests can fail (one mutation each, restored after)**

1. In `box_mesh`, swap `+x` face to `1, 7, 3, 1, 5, 7`: `a_box_mesh_is_closed_and_wound_outward` must fail on volume.
2. In `merge`, drop `+ base`: `merging_offsets...` must fail.
3. In `validate`, change the range check to `>` instead of `>=`: the "out of range" case must fail (index 8 would pass).

Record the failure text for each in the commit body.

- [ ] **Step 6: Commit**

```bash
just check
git add crates/elements-ember/src/mesh.rs crates/elements-ember/src/lib.rs crates/elements-ember/tests/mesh.rs
git commit -m "Add a validated triangle mesh type"
```

---

### Task 3: Procedural shack and the FT0 shot spec

**Files:**
- Create: `crates/elements-ember/src/shack.rs`
- Modify: `crates/elements-ember/src/lib.rs` (add `pub mod shack;`)
- Test: `crates/elements-ember/tests/shack.rs`
- Create: `docs/superpowers/specs/2026-09-30-flamethrower-shot-ft0.md`

**Interfaces:**
- Consumes: `Mesh::{box_mesh, merge, bounds}` (Task 2).
- Produces: `pub struct ShackParams { pub size: [f32; 3], pub plank_height: f32, pub thickness: f32, pub gap: f32, pub broken_fraction: f32, pub seed: u64 }`; `pub fn shack(p: &ShackParams) -> Mesh` (Task 8 and FT6 use it). The shack stands on z = 0 with its footprint `size[0]` × `size[1]` centred on the origin in x/y; the caller translates it with the collider `Transform`.

- [ ] **Step 1: Write the failing tests** (`tests/shack.rs`)

```rust
use elements_ember::shack::{ShackParams, shack};

fn params() -> ShackParams {
    ShackParams {
        size: [1.2, 1.0, 1.0],
        plank_height: 0.1,
        thickness: 0.02,
        gap: 0.01,
        broken_fraction: 0.0,
        seed: 7,
    }
}

#[test]
fn an_unbroken_shack_has_every_plank_and_stays_inside_its_size() {
    let m = shack(&params());
    m.validate("test").unwrap();
    // 4 walls × floor(height / (plank + gap)) planks + 1 roof slab, 12 triangles each.
    let rows = (1.0f32 / 0.11).floor() as usize; // 9
    assert_eq!(m.triangle_count(), (4 * rows + 1) * 12);
    let (lo, hi) = m.bounds();
    let eps = 1e-5;
    assert!(lo[0] >= -0.6 - eps && hi[0] <= 0.6 + eps, "x {lo:?} {hi:?}");
    assert!(lo[1] >= -0.5 - eps && hi[1] <= 0.5 + eps, "y {lo:?} {hi:?}");
    assert!(lo[2] >= -eps && hi[2] <= 1.0 + 0.02 + eps, "z {lo:?} {hi:?}");
}

#[test]
fn breakage_removes_planks_deterministically_per_seed() {
    let p = |seed, f| ShackParams { seed, broken_fraction: f, ..params() };
    let full = shack(&p(1, 0.0)).triangle_count();
    let a = shack(&p(1, 0.3));
    let b = shack(&p(1, 0.3));
    let c = shack(&p(2, 0.3));
    assert_eq!(a, b, "same seed, same shack");
    assert!(a.triangle_count() < full, "some planks are gone");
    assert_ne!(a, c, "a different seed breaks different planks");
    assert!(shack(&p(1, 1.0)).triangle_count() >= 12, "the roof always remains");
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo nextest run -p elements-ember --test shack`
Expected: FAIL to compile (`unresolved import elements_ember::shack`).

- [ ] **Step 3: Implement `src/shack.rs`**

```rust
//! A procedural plank shack for the flamethrower shot (roadmap spec §4, FT0):
//! four walls of horizontal planks with gaps, some planks removed, and a roof
//! slab. Our own geometry; it does not copy any reference asset.

use crate::mesh::Mesh;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShackParams {
    /// Footprint x, footprint y, wall height, metres.
    pub size: [f32; 3],
    pub plank_height: f32,
    pub thickness: f32,
    /// Vertical gap between plank rows, metres.
    pub gap: f32,
    /// Fraction of wall planks removed, 0 to 1.
    pub broken_fraction: f32,
    pub seed: u64,
}

/// SplitMix64: a small explicit-seed generator, so breakage is reproducible.
fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub fn shack(p: &ShackParams) -> Mesh {
    let [w, d, h] = p.size;
    let (hw, hd, t) = (w / 2.0, d / 2.0, p.thickness);
    let rows = (h / (p.plank_height + p.gap)).floor() as usize;
    let mut rng = p.seed;
    let mut mesh = Mesh {
        positions: vec![],
        indices: vec![],
    };
    // The roof always remains, so the mesh is never empty.
    mesh.merge(&Mesh::box_mesh(
        [-hw - t, -hd - t, h],
        [hw + t, hd + t, h + t],
    ));
    for row in 0..rows {
        let z0 = row as f32 * (p.plank_height + p.gap);
        let z1 = z0 + p.plank_height;
        // South, north, west, east walls; the side walls span the full depth
        // and the front/back walls sit between them so planks do not overlap.
        let walls = [
            ([-hw, -hd, z0], [hw, -hd + t, z1]),
            ([-hw, hd - t, z0], [hw, hd, z1]),
            ([-hw, -hd + t, z0], [-hw + t, hd - t, z1]),
            ([hw - t, -hd + t, z0], [hw, hd - t, z1]),
        ];
        for (lo, hi) in walls {
            let roll = (next(&mut rng) >> 11) as f64 / (1u64 << 53) as f64;
            if roll >= f64::from(p.broken_fraction) {
                mesh.merge(&Mesh::box_mesh(lo, hi));
            }
        }
    }
    mesh
}
```

Add `pub mod shack;` to `lib.rs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo nextest run -p elements-ember --test shack`
Expected: 2 passed. The test's x and y bounds hold because walls span exactly `±size/2`; the roof overhangs by `thickness`, so if the bounds assertion fails on the roof, widen the test's bound by `thickness` rather than the geometry, and note why.

- [ ] **Step 5: Prove the tests can fail**

1. Change `rows` to `rows + 1`: the unbroken count test must fail.
2. Replace `next(&mut rng)` with a constant `0`: `breakage_removes...` must fail (nothing ever breaks).

- [ ] **Step 6: Write the FT0 shot spec**

Create `docs/superpowers/specs/2026-09-30-flamethrower-shot-ft0.md` with exactly these sections and decided values (edit numbers only where Step 7's measurements require, and say so in the doc):

```markdown
# Flamethrower vs. shack: shot spec (FT0)

**Date:** 2026-09-30. **Roadmap:** `2026-09-30-flamethrower-napalm-roadmap-design.md` §4.

## Reference
Jason Key (@key_vfx), "flamethower vs. shack", 2026-09-01, EmberGen 1.2.11, 30 s. Used for
comparison only; no frames or assets are stored in this repository.

## Scene (Ember units, metres, z up)
- Domain `dims [256, 128, 128]`, `domain_size 4.0` (x longest): x 4.0 m, y 2.0 m, z 2.0 m,
  dx = 15.625 mm. Non-cubic support: <Task 1 result, verbatim>.
- Shack: `ShackParams { size [1.2, 1.0, 1.0], plank_height 0.1, thickness 0.02, gap 0.01,
  broken_fraction 0.2, seed 7 }`, collider origin at (2.9, 1.0, 0.0). Planks are thinner than
  one voxel, so the mesh collider uses `offset = 0.5 * dx`.
- Nozzle: cone along +x, `length 0.3`, `radius_start 0.03`, `radius_end 0.06`, origin at
  (0.4, 1.0, 0.6), emitting `fuel_rate`, `temperature_rate` and a local-frame velocity of
  `[14, 0, 0]` m/s with `velocity_blend 20`; noise amplitude 0.6, scale 0.08 m; pulse on
  frames 10–150.
- Frames 1–240 at 24 fps. Camera: side view, orthographic-like long lens framing the jet and
  the shack; set in FT6.

## Acceptance for the shot (FT6)
1. Jet visibly reaches the shack and wraps it; the shack ignites (FT4) and keeps burning.
2. Smoke and flame separate (flame output, density output, fuel output all available).
3. No NaN, no domain-edge blow-up; divergence recorded per FT3.
4. Verdict versus the reference, including detail and speed gaps, recorded honestly.

## Budget
<filled in by Task 3 Step 7: bake frame time and peak memory at [256,128,128], machine load>
```

- [ ] **Step 7: Measure the budget and fill it in**

Run the existing fire path at the shot's dims to learn whether the domain fits: copy `examples/plume.elements` to the scratchpad directory, set `"dims": [256, 128, 128]`, `"domain_size": 4.0`, move the emitter to `[0.5, 1.0, 0.6]`, and bake 10 frames with the CLI (`cargo run --release -p elements-cli -- --help` shows the exact subcommand; use it). Record wall time per frame, the machine load (`uptime`), and peak memory (`/usr/bin/time -l`). Write them in the spec's Budget section. If it runs out of GPU memory (risk g), record that and reduce to `[192, 96, 96]`, then update the Scene section to match.

- [ ] **Step 8: Commit**

```bash
just check
git add crates/elements-ember/src/shack.rs crates/elements-ember/src/lib.rs crates/elements-ember/tests/shack.rs docs/superpowers/specs/2026-09-30-flamethrower-shot-ft0.md
git commit -m "Add a procedural shack and the flamethrower shot spec"
```

---

### Task 4: A cone shape (FT1)

**Files:**
- Modify: `crates/elements-ember/src/transform.rs` (`Shape`, `validate`, `ShapeGpu::new`)
- Modify: `crates/elements-ember/src/kernels/shaders/shape.wgsl`
- Modify: `crates/elements-ember/src/bench/mod.rs:216-222` (the exhaustive `match collider.shape`)
- Test: `crates/elements-ember/tests/cone.rs`

**Interfaces:**
- Produces: `Shape::Cone { length: f32, radius_start: f32, radius_end: f32 }` (JSON `{"cone": {"length":…, "radius_start":…, "radius_end":…}}` via the existing lowercase serde rename). Axis is the shape's local +x; centred at its local origin; `radius_start` is the −x end, `radius_end` the +x end. `ShapeGpu` kind 2, extents `[length/2, radius_start, radius_end]`.
- Consumes: `fill_emitter`/`fill_collider` (unchanged).

- [ ] **Step 1: Write the failing tests** (`tests/cone.rs`)

```rust
mod common;

use common::*;
use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, PipelineCache};
use elements_ember::collider::{ColliderFields, ColliderParams, fill_collider};
use elements_ember::transform::{Key, Rotate, Shape, Transform};

const SPF: f64 = 1.0 / 24.0;

/// SDF of `shape` at every cell centre of a 32³, 2 m domain.
fn sdf(shape: Shape, rotate: Option<Rotate>, at: [f32; 3]) -> Vec<f32> {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(32, 32, 32);
    let out = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    let v = pool.acquire_staggered_uninit(&gpu, cells).unwrap();
    let params = ColliderParams {
        shape,
        transform: Transform {
            keys: vec![Key { frame: 0.0, translate: at, rotate }],
        },
    };
    let pose = params.transform.pose(0.0, SPF);
    fill_collider(
        &gpu,
        &mut cache,
        &params,
        &pose,
        2.0 / 32.0,
        ColliderFields { sdf: &out, velocity: &v },
    )
    .unwrap();
    out.read_back(&gpu).unwrap()
}

fn centre(i: u32) -> f32 {
    (i as f32 + 0.5) * (2.0 / 32.0)
}

/// With equal radii the cone is a cylinder along x, length 0.5, radius 0.2.
/// On its axis, inside, the distance is minus the nearer of the cap (0.25 − |x|)
/// and the side (0.2); outside the cap on the axis it is the gap to the cap.
#[test]
fn a_cylinder_cone_has_the_analytic_distance_on_its_axis() {
    let s = sdf(
        Shape::Cone { length: 0.5, radius_start: 0.2, radius_end: 0.2 },
        None,
        [1.0, 1.0, 1.0],
    );
    let cells = FieldDims::new(32, 32, 32);
    // Axis cells sit at y = z = 1.03125 (cell 16), 0.03125 m off the true axis,
    // so compare radially: distance to the side is 0.2 − 0.03125·√2.
    let off = (2.0f32 * 0.03125f32 * 0.03125).sqrt();
    let inside = s[index(cells, 16, 16, 16)]; // x = 1.03125, 0.03125 from the centre
    let expected_inside = -(0.2 - off).min(0.25 - 0.03125);
    assert!((inside - expected_inside).abs() < 1e-5, "{inside} vs {expected_inside}");
    // Cell 24 is x = 1.53125: 0.53125 from the centre, 0.28125 beyond the cap.
    let past = s[index(cells, 24, 16, 16)];
    let gap = 0.53125 - 0.25;
    assert!((past - gap).abs() < 1e-3, "{past} vs {gap}");
}

/// A true cone (radius_end 0) points at its tip: on the axis beyond the tip
/// the distance is the gap to the tip.
#[test]
fn a_true_cone_measures_to_its_tip() {
    let s = sdf(
        Shape::Cone { length: 0.5, radius_start: 0.2, radius_end: 0.0 },
        None,
        [1.0, 1.0, 1.0],
    );
    let cells = FieldDims::new(32, 32, 32);
    let past = s[index(cells, 24, 16, 16)]; // 0.28125 beyond the tip at 0.25
    assert!((past - 0.28125).abs() < 1e-2, "{past}");
    assert!(s[index(cells, 16, 16, 16)] < 0.0, "the middle is inside");
}

/// A cone turned 90° about z points along +y.
#[test]
fn rotation_turns_the_cone() {
    let s = sdf(
        Shape::Cone { length: 0.8, radius_start: 0.1, radius_end: 0.1 },
        Some(Rotate { axis: [0.0, 0.0, 1.0], degrees: 90.0 }),
        [1.0, 1.0, 1.0],
    );
    let cells = FieldDims::new(32, 32, 32);
    assert!(s[index(cells, 16, 22, 16)] < 0.0, "along +y is inside");
    assert!(s[index(cells, 22, 16, 16)] > 0.0, "along +x is outside");
}

#[test]
fn bad_cones_are_rejected() {
    for (what, shape) in [
        ("zero length", Shape::Cone { length: 0.0, radius_start: 0.1, radius_end: 0.1 }),
        ("negative radius", Shape::Cone { length: 1.0, radius_start: -0.1, radius_end: 0.1 }),
        ("both radii zero", Shape::Cone { length: 1.0, radius_start: 0.0, radius_end: 0.0 }),
        ("nan", Shape::Cone { length: f32::NAN, radius_start: 0.1, radius_end: 0.1 }),
    ] {
        assert!(shape.validate("ember.emitter").is_err(), "{what}");
    }
    Shape::Cone { length: 1.0, radius_start: 0.0, radius_end: 0.1 }
        .validate("ember.emitter")
        .unwrap();
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo nextest run -p elements-ember --test cone`
Expected: FAIL to compile (no variant `Cone`).

- [ ] **Step 3: Implement**

`transform.rs`: add the variant, validation and GPU packing.

```rust
pub enum Shape {
    Sphere { radius: f32 },
    Box { half_extents: [f32; 3] },
    /// A truncated cone along the local +x axis, centred on the origin:
    /// `radius_start` at the −x end, `radius_end` at the +x end.
    Cone { length: f32, radius_start: f32, radius_end: f32 },
}
```

In `Shape::validate` add:

```rust
Self::Cone { length, radius_start, radius_end } => {
    params::finite(kind, "cone", &[*length, *radius_start, *radius_end])?;
    if *length <= 0.0 {
        return Err(params::bad(kind, format!("cone length must be positive, got {length}")));
    }
    if *radius_start < 0.0 || *radius_end < 0.0 {
        return Err(params::bad(
            kind,
            format!("cone radii must not be negative, got {radius_start} and {radius_end}"),
        ));
    }
    if *radius_start == 0.0 && *radius_end == 0.0 {
        return Err(params::bad(kind, "a cone needs at least one nonzero radius"));
    }
}
```

In `ShapeGpu::new` match: `Shape::Cone { length, radius_start, radius_end } => (2, [length / 2.0, radius_start, radius_end]),`. Update the `ShapeGpu` doc comment ("kind 0 sphere, 1 box, 2 cone").

`shape.wgsl`: update the `kind` comment to `0 sphere, 1 box, 2 cone`; `extents` comment to add `cone: half length, radius at −x, radius at +x`; and in `shape_sdf`, after the sphere branch:

```wgsl
    if (shape.kind == 2u) {
        // Capped cone (Inigo Quilez's exact formula) with its axis along local x.
        let h = shape.extents.x;
        let r1 = shape.extents.y;   // radius at -x
        let r2 = shape.extents.z;   // radius at +x
        let q = vec2<f32>(length(p.yz), p.x);
        let k1 = vec2<f32>(r2, h);
        let k2 = vec2<f32>(r2 - r1, 2.0 * h);
        let ca = vec2<f32>(q.x - min(q.x, select(r2, r1, q.y < 0.0)), abs(q.y) - h);
        let t = clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
        let cb = q - k1 + k2 * t;
        let s = select(1.0, -1.0, cb.x < 0.0 && ca.y < 0.0);
        return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
    }
```

`bench/mod.rs`: add `Shape::Cone { .. } => panic!("bench colliders are spheres or boxes"),` to the `match collider.shape` (the assert-style panic matches that function's "bench colliders are static/unrotated" convention).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo nextest run -p elements-ember --test cone`
Expected: 4 passed. If the cylinder test is off, recompute the expected values from the cell centres before touching the kernel; the shader formula is the published exact one.

- [ ] **Step 5: Prove the tests can fail**

1. In `shape.wgsl` swap `r1`/`r2` in `ca` (`select(r1, r2, …)`): `a_true_cone_measures_to_its_tip` must fail or `rotation` must be unaffected, and at least one test must fail. Record which.
2. In `ShapeGpu::new` pack `[length, …]` instead of `length / 2.0`: `a_cylinder_cone_has_the_analytic_distance_on_its_axis` must fail.

- [ ] **Step 6: Commit**

```bash
just check
git add crates/elements-ember/src/transform.rs crates/elements-ember/src/kernels/shaders/shape.wgsl crates/elements-ember/src/bench/mod.rs crates/elements-ember/tests/cone.rs
git commit -m "Add a cone shape for nozzles"
```

---

### Task 5: Local-frame jet velocity (FT1)

**Files:**
- Modify: `crates/elements-ember/src/shape_emitter.rs` (`EmitterParams`, `EmitterGpu`)
- Modify: `crates/elements-ember/src/kernels/shaders/emitter_common.wgsl`, `emitter_faces.wgsl`
- Test: `crates/elements-ember/tests/jet.rs`

**Interfaces:**
- Consumes: `Shape::Cone` (Task 4); `fill_emitter(gpu, cache, params, pose, seconds, dx, EmitterFields)`.
- Produces: `EmitterParams.velocity_local: bool` (`#[serde(default)]`, JSON key `velocity_local`, default `false`). When true, `velocity` is in the emitter's local frame and is rotated into world space by the pose before use. `EmitterGpu::_pad` first word becomes `velocity_local: u32`; the struct stays 80 bytes.

- [ ] **Step 1: Write the failing tests** (`tests/jet.rs`)

```rust
mod common;

use common::*;
use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, PipelineCache};
use elements_ember::shape_emitter::{EmitterFields, EmitterParams, fill_emitter};
use elements_ember::transform::{Key, Rotate, Shape, Transform};

const SPF: f64 = 1.0 / 24.0;

/// Face velocities [u, v, w] of one nozzle emitter at `frame`, and its density.
fn run(params: &EmitterParams, frame: f64) -> ([Vec<f32>; 3], Vec<f32>) {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(32, 32, 32);
    let f: [_; 3] =
        std::array::from_fn(|_| pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap());
    let v = pool.acquire_staggered_uninit(&gpu, cells).unwrap();
    let pose = params.transform.pose(frame, SPF);
    fill_emitter(
        &gpu,
        &mut cache,
        params,
        &pose,
        frame * SPF,
        2.0 / 32.0,
        EmitterFields { density: &f[0], temperature: &f[1], weight: &f[2], velocity: &v },
    )
    .unwrap();
    (read_staggered(&gpu, &v), f[0].read_back(&gpu).unwrap())
}

fn nozzle(rotate: Option<Rotate>, local: bool) -> EmitterParams {
    EmitterParams {
        density_rate: 1.0,
        velocity: [14.0, 0.0, 0.0],
        velocity_blend: 20.0,
        velocity_local: local,
        ..EmitterParams::new(
            Shape::Cone { length: 0.6, radius_start: 0.1, radius_end: 0.2 },
            Transform {
                keys: vec![Key { frame: 0.0, translate: [1.0, 1.0, 1.0], rotate }],
            },
        )
    }
}

fn turn_z_90() -> Option<Rotate> {
    Some(Rotate { axis: [0.0, 0.0, 1.0], degrees: 90.0 })
}

/// A local-frame velocity follows the nozzle: turned 90° about z, the +x jet
/// points along +y, and no momentum goes along x.
#[test]
fn a_rotated_nozzle_pushes_along_its_own_axis() {
    let ([u, v, w], _) = run(&nozzle(turn_z_90(), true), 0.0);
    let (nx, ny) = (33usize, 33usize);
    // The u-face at cell (16,16,16) and the v-face at (16,16,16): inside the cone.
    let ui = 16 + nx * (16 + 32 * 16);
    let vi = 16 + 32 * (16 + ny * 16);
    assert!(v[vi].abs() > 13.9 && v[vi].abs() < 14.1, "v {}", v[vi]);
    assert!(u[ui].abs() < 1e-4, "u {}", u[ui]);
    assert!(w.iter().all(|x| x.abs() < 1e-4), "no z momentum");
}

/// Without `velocity_local`, the velocity stays in world space even when the
/// nozzle turns, which is the behaviour every existing document relies on.
#[test]
fn a_world_frame_velocity_ignores_the_rotation() {
    let ([u, v, _], _) = run(&nozzle(turn_z_90(), false), 0.0);
    let (nx, ny) = (33usize, 33usize);
    let ui = 16 + nx * (16 + 32 * 16);
    let vi = 16 + 32 * (16 + ny * 16);
    assert!((u[ui] - 14.0).abs() < 0.1, "u {}", u[ui]);
    assert!(v[vi].abs() < 1e-4, "v {}", v[vi]);
}
```

Off-frame silence is node-level and shape-agnostic, and `tests/shape_emitter.rs::an_emitter_is_silent_outside_its_active_frames` already covers it; no new test is needed for it.

- [ ] **Step 2: Run to verify failure**

Run: `cargo nextest run -p elements-ember --test jet`
Expected: FAIL to compile (no field `velocity_local`).

- [ ] **Step 3: Implement**

`EmitterParams`: add
```rust
    /// Whether `velocity` is in the emitter's local frame (a nozzle's axis)
    /// instead of world space. Default false.
    #[serde(default)]
    pub velocity_local: bool,
```
and `velocity_local: false` in `EmitterParams::new`. `EmitterGpu`: replace `_pad: [u32; 2]` with `velocity_local: u32, _pad: u32` and set `velocity_local: u32::from(p.velocity_local), _pad: 0` in `EmitterGpu::new` (the size assertion stays 80). In `emitter_common.wgsl` replace `_pad1: u32,` with `velocity_local: u32,`.

In `emitter_faces.wgsl` replace the velocity line:
```wgsl
    // A local-frame velocity turns with the emitter. `world_to_local` is the
    // transpose of local-to-world, so `v * M` is `transpose(M) * v`.
    var target_velocity = emitter.velocity;
    if (emitter.velocity_local != 0u) {
        target_velocity = emitter.velocity * shape.world_to_local;
    }
    let v = target_velocity + shape_velocity(x);
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo nextest run -p elements-ember --test jet --test shape_emitter --test emitter`
Expected: all pass, existing emitter tests unchanged (defaults keep old behaviour).

- [ ] **Step 5: Prove the tests can fail**

1. In `emitter_faces.wgsl` multiply `shape.world_to_local * emitter.velocity` (the transpose mistake): `a_rotated_nozzle_pushes_along_its_own_axis` must fail (momentum goes to −y).
2. Force the branch off (`!= 0u` to `== 7u`): the same test must fail and `a_world_frame...` must still pass.

- [ ] **Step 6: Commit**

```bash
just check
git add crates/elements-ember/src crates/elements-ember/tests/jet.rs
git commit -m "Let a nozzle's jet velocity follow its orientation"
```

---

### Task 6: Mesh collider node (FT2)

**Files:**
- Create: `crates/elements-ember/src/mesh_collider.rs`
- Create: `crates/elements-ember/src/kernels/shaders/mesh_sdf.wgsl`
- Modify: `crates/elements-ember/src/kernels/mod.rs` (add `storage_buffer`)
- Modify: `crates/elements-ember/src/lib.rs` (`pub mod mesh_collider;` and `registry.register(mesh_collider::KIND, mesh_collider::build);`)
- Test: `crates/elements-ember/tests/mesh_collider.rs`

**Interfaces:**
- Consumes: `Mesh` (Task 2); `ColliderFields { sdf, velocity }` and the collider's face kernel (`collider_common.wgsl` + `shape.wgsl` + `collider_faces.wgsl`, as `collider.rs` concatenates them); `Transform`/`Pose`/`ShapeGpu`; `produce`.
- Produces (used by Tasks 7–8 and FT4–FT6):
  - `pub const KIND: &str = "ember.mesh_collider";`
  - `pub struct MeshColliderParams { pub mesh: Mesh, pub transform: Transform, pub offset: f32 }` (`offset` `#[serde(default)]`, metres ≥ 0; **subtracted from the signed distance**, inflating the solid).
  - `pub fn fill_mesh_collider(gpu, cache, params: &MeshColliderParams, pose: &Pose, dx: f32, out: ColliderFields<'_>) -> Result<(), GpuError>`
  - Node sockets: outputs `[Field, VectorField]`, identical to `ember.collider`, so `ember.collider_union` and the solver's collider inputs accept it unchanged.
  - `pub(crate) fn storage_buffer(gpu, label, bytes) -> Result<wgpu::Buffer, GpuError>` in `kernels/mod.rs`.

- [ ] **Step 1: Write the failing tests** (`tests/mesh_collider.rs`)

```rust
mod common;

use common::*;
use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, PipelineCache};
use elements_ember::collider::ColliderFields;
use elements_ember::mesh::Mesh;
use elements_ember::mesh_collider::{MeshColliderParams, fill_mesh_collider};
use elements_ember::transform::{Key, Rotate, Transform};

const SPF: f64 = 1.0 / 24.0;
const N: u32 = 32;
const DX: f32 = 2.0 / 32.0;

fn fill(params: &MeshColliderParams, frame: f64) -> (Vec<f32>, [Vec<f32>; 3]) {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(N, N, N);
    let sdf = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    let v = pool.acquire_staggered_uninit(&gpu, cells).unwrap();
    let pose = params.transform.pose(frame, SPF);
    fill_mesh_collider(&gpu, &mut cache, params, &pose, DX, ColliderFields { sdf: &sdf, velocity: &v })
        .unwrap();
    (sdf.read_back(&gpu).unwrap(), read_staggered(&gpu, &v))
}

fn at(translate: [f32; 3], rotate: Option<Rotate>) -> Transform {
    Transform { keys: vec![Key { frame: 0.0, translate, rotate }] }
}

fn cell(i: u32) -> f64 {
    (f64::from(i) + 0.5) * f64::from(DX)
}

/// Exact distance to the axis-aligned box [lo, hi], negative inside.
fn box_sdf(lo: [f64; 3], hi: [f64; 3], p: [f64; 3]) -> f64 {
    let c: [f64; 3] = std::array::from_fn(|a| (lo[a] + hi[a]) / 2.0);
    let h: [f64; 3] = std::array::from_fn(|a| (hi[a] - lo[a]) / 2.0);
    let q: [f64; 3] = std::array::from_fn(|a| (p[a] - c[a]).abs() - h[a]);
    let out = q.iter().map(|v| v.max(0.0).powi(2)).sum::<f64>().sqrt();
    out + q[0].max(q[1]).max(q[2]).min(0.0)
}

/// A box mesh gives the exact box distance, inside and out, at every cell.
#[test]
fn a_box_mesh_matches_the_analytic_box_distance() {
    let params = MeshColliderParams {
        mesh: Mesh::box_mesh([-0.3, -0.2, -0.25], [0.3, 0.2, 0.25]),
        transform: at([1.0, 1.0, 1.0], None),
        offset: 0.0,
    };
    let (sdf, _) = fill(&params, 0.0);
    let (lo, hi) = ([0.7, 0.8, 0.75], [1.3, 1.2, 1.25]);
    let cells = FieldDims::new(N, N, N);
    let mut worst = 0.0f64;
    for k in 0..N {
        for j in 0..N {
            for i in 0..N {
                let want = box_sdf(lo, hi, [cell(i), cell(j), cell(k)]);
                let got = f64::from(sdf[index(cells, i, j, k)]);
                worst = worst.max((got - want).abs());
            }
        }
    }
    assert!(worst < 1e-4, "worst error {worst} m");
}

/// A rotated mesh is the box turned: the axis-aligned check on a box turned
/// 90° about z swaps its x and y half-extents.
#[test]
fn a_rotated_mesh_is_the_rotated_shape() {
    let params = MeshColliderParams {
        mesh: Mesh::box_mesh([-0.5, -0.1, -0.1], [0.5, 0.1, 0.1]),
        transform: at([1.0, 1.0, 1.0], Some(Rotate { axis: [0.0, 0.0, 1.0], degrees: 90.0 })),
        offset: 0.0,
    };
    let (sdf, _) = fill(&params, 0.0);
    let cells = FieldDims::new(N, N, N);
    assert!(sdf[index(cells, 16, 22, 16)] < 0.0, "along +y is inside");
    assert!(sdf[index(cells, 22, 16, 16)] > 0.0, "along +x is outside");
}

/// `offset` inflates the solid by exactly that distance.
#[test]
fn offset_shrinks_the_distance() {
    let base = MeshColliderParams {
        mesh: Mesh::box_mesh([-0.3; 3], [0.3; 3]),
        transform: at([1.0; 3], None),
        offset: 0.0,
    };
    let inflated = MeshColliderParams { offset: 0.05, ..base.clone() };
    let (a, _) = fill(&base, 0.0);
    let (b, _) = fill(&inflated, 0.0);
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y - 0.05).abs() < 1e-5, "{x} {y}");
    }
}

/// A zero-area triangle inside a closed mesh is skipped, not turned into NaN.
#[test]
fn a_degenerate_triangle_does_not_poison_the_field() {
    let mut mesh = Mesh::box_mesh([-0.3; 3], [0.3; 3]);
    mesh.positions.push([0.1, 0.1, 0.1]);
    let v = (mesh.positions.len() - 1) as u32;
    mesh.indices.extend_from_slice(&[v, v, v]);
    let params = MeshColliderParams { mesh, transform: at([1.0; 3], None), offset: 0.0 };
    let (sdf, _) = fill(&params, 0.0);
    assert!(sdf.iter().all(|d| d.is_finite()));
    let cells = FieldDims::new(N, N, N);
    assert!(sdf[index(cells, 16, 16, 16)] < 0.0);
}

/// A moving mesh carries its material velocity to the faces, as
/// `ember.collider` does.
#[test]
fn a_moving_mesh_has_its_linear_velocity_on_the_faces() {
    let mut t = at([1.0, 1.0, 1.0], None);
    t.keys.push(Key { frame: 24.0, translate: [2.0, 1.0, 1.0], rotate: None });
    let params = MeshColliderParams { mesh: Mesh::box_mesh([-0.3; 3], [0.3; 3]), transform: t, offset: 0.0 };
    let (_, [u, v, _]) = fill(&params, 12.0);
    assert!((u[0] - 1.0 / (24.0 * SPF) as f32).abs() < 1e-4, "u {}", u[0]);
    assert!(v[0].abs() < 1e-5);
}
```

Also add to the same file a document-level test that the node is registered and rejects a bad mesh:

```rust
#[test]
fn the_node_is_registered_and_validates_its_mesh() {
    use elements_core::graph::Document;
    let ok = r#"{ "version": 3, "dims": [8,8,8], "fps": 24.0, "start_frame": 1, "domain_size": 2.0,
      "nodes": [ { "id": 0, "kind": "ember.mesh_collider",
        "params": { "mesh": { "positions": [[0,0,0],[1,0,0],[0,1,0]], "indices": [0,1,2] },
                    "transform": { "keys": [ { "frame": 0.0, "translate": [1,1,1] } ] } } },
        { "id": 1, "kind": "core.output", "params": {} } ],
      "edges": [ { "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 } ], "output": 1 }"#;
    Document::from_json(ok).unwrap().into_graph(&elements_ember::registry()).unwrap();
    let bad = ok.replace("\"indices\": [0,1,2]", "\"indices\": [0,1,9]");
    let err = Document::from_json(&bad).unwrap().into_graph(&elements_ember::registry());
    assert!(err.is_err(), "an out-of-range index must be rejected at build");
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo nextest run -p elements-ember --test mesh_collider`
Expected: FAIL to compile (`unresolved import elements_ember::mesh_collider`).

- [ ] **Step 3: Add the storage-buffer helper** to `kernels/mod.rs` next to `uniform_buffer`:

```rust
/// A read-only storage buffer holding `bytes`, created inside an error scope.
pub(crate) fn storage_buffer(
    gpu: &GpuContext,
    label: &str,
    bytes: &[u8],
) -> Result<wgpu::Buffer, GpuError> {
    gpu.scoped(|| {
        gpu.device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            })
    })
}
```

Check `GpuContext::required_limits`/`max_storage_buffer_binding_size`: the largest buffer is `indices` at `4 × 3 × 1e6 = 12 MB` and `positions` at 16 B per vertex, well under the 128 MiB `downlevel_defaults` storage binding limit (the cap `MAX_TRIANGLES` exists to keep it so). Verify by reading `gpu::required_limits` and note the number in the commit body.

- [ ] **Step 4: Write the kernel** `mesh_sdf.wgsl`

```wgsl
// ember.mesh_collider, cell pass: the signed distance from every cell centre
// to a triangle mesh, metres, negative inside. Brute force: every cell tests
// every triangle, so cost is cells × triangles. Concatenated after
// `shape.wgsl`, which supplies `Shape` (used here only for the pose).

struct MeshParams {
    dims: vec3<u32>,
    dx: f32,
    tri_count: u32,
    offset: f32,     // subtracted from the distance: inflates the solid
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var sdf: texture_storage_3d<r32float, write>;
@group(0) @binding(1) var<uniform> mesh: MeshParams;
@group(0) @binding(2) var<uniform> shape: Shape;
@group(0) @binding(3) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> tris: array<u32>;

// Ericson, Real-Time Collision Detection §5.1.5: the closest point on
// triangle abc to p.
fn closest_on_triangle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { return a; }
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { return b; }
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) { return a + ab * (d1 / (d1 - d3)); }
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { return c; }
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) { return a + ac * (d2 / (d2 - d6)); }
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = 1.0 / (va + vb + vc);
    return a + ab * (vb * denom) + ac * (vc * denom);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= mesh.dims)) {
        return;
    }
    let world = (vec3<f32>(gid) + vec3<f32>(0.5)) * mesh.dx;
    let p = shape.world_to_local * (world - shape.origin);

    var best = 1.0e30;
    var best_abs_dot = 0.0;
    var sign = 1.0;
    for (var t = 0u; t < mesh.tri_count; t = t + 1u) {
        let a = vertices[tris[3u * t]].xyz;
        let b = vertices[tris[3u * t + 1u]].xyz;
        let c = vertices[tris[3u * t + 2u]].xyz;
        let n = cross(b - a, c - a);
        let nl = length(n);
        if (nl <= 0.0) {
            continue;   // a zero-area triangle has no normal and no surface
        }
        let d = p - closest_on_triangle(p, a, b, c);
        let dist = length(d);
        let ndot = dot(d, n / nl);
        // The nearest triangle's normal gives the side. Where two triangles
        // tie (an edge or vertex), the one more aligned with the offset wins.
        let better = dist < best - 1.0e-6;
        let tie = !better && dist <= best + 1.0e-6 && abs(ndot) > best_abs_dot;
        if (better || tie) {
            best = min(best, dist);
            best_abs_dot = abs(ndot);
            sign = select(1.0, -1.0, ndot < 0.0);
        }
    }
    textureStore(sdf, vec3<i32>(gid), vec4<f32>(sign * best - mesh.offset, 0.0, 0.0, 0.0));
}
```

- [ ] **Step 5: Write the node** `mesh_collider.rs`. Mirror `collider.rs` closely (same faces pass, same `produce` call, same `build`), changing only the cell pass:

```rust
//! `ember.mesh_collider`: a triangle mesh the fluid cannot enter, outputting
//! the same signed-distance and face-velocity pair as `ember.collider`
//! (flamethrower roadmap spec §4, FT2).

use elements_core::gpu::{
    Axis, ComputeBatch, GpuContext, GpuError, PipelineCache,
};
use elements_core::graph::{DocError, EvalCtx, Node, NodeError, SocketSpec, SocketType, Value};
use serde::{Deserialize, Serialize};

use crate::collider::ColliderFields;
use crate::kernels::{Bind, axis_index, bind_group, storage_buffer, uniform_buffer};
use crate::mesh::Mesh;
use crate::node_util::produce;
use crate::params;
use crate::transform::{Pose, Shape, ShapeGpu, Transform};

pub const KIND: &str = "ember.mesh_collider";

const CELLS_WGSL: &str = concat!(
    include_str!("kernels/shaders/shape.wgsl"),
    include_str!("kernels/shaders/mesh_sdf.wgsl"),
);

// The face pass is the collider's: only the pose velocity is read from
// `shape`, so the shape kind is irrelevant.
const FACES_WGSL: &str = concat!(
    include_str!("kernels/shaders/collider_common.wgsl"),
    include_str!("kernels/shaders/shape.wgsl"),
    include_str!("kernels/shaders/collider_faces.wgsl"),
);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshColliderParams {
    pub mesh: Mesh,
    pub transform: Transform,
    /// Metres subtracted from the signed distance: inflates the solid so
    /// planks thinner than a voxel still block flow. Zero or more.
    #[serde(default)]
    pub offset: f32,
}

/// Matches `MeshParams` in mesh_sdf.wgsl, 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshGpu {
    dims: [u32; 3],
    dx: f32,
    tri_count: u32,
    offset: f32,
    _pad: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<MeshGpu>() == 32);

/// Matches `ColliderGpu` in collider.rs (the shared face pass), 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FaceGpu {
    dims: [u32; 3],
    dx: f32,
    axis: u32,
    _pad: [u32; 3],
}

/// Write `params`' SDF and velocity at `pose` into `out`. Submits its own batch.
pub fn fill_mesh_collider(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    params: &MeshColliderParams,
    pose: &Pose,
    dx: f32,
    out: ColliderFields<'_>,
) -> Result<(), GpuError> {
    out.check("fill_mesh_collider")?;
    let cells = out.sdf.dims();
    let dims = [cells.x, cells.y, cells.z];
    let cell_pipe = cache.get_or_create(gpu, "ember.mesh_collider.cells", CELLS_WGSL, "main")?;
    let face_pipe = cache.get_or_create(gpu, "ember.mesh_collider.faces", FACES_WGSL, "main")?;
    // `Shape` supplies the pose; the kind and extents are unused.
    let shape = uniform_buffer(
        gpu,
        "ember-shape",
        bytemuck::bytes_of(&ShapeGpu::new(&Shape::Sphere { radius: 1.0 }, pose)),
    )?;
    let vertices: Vec<[f32; 4]> = params.mesh.positions.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
    let vertex_buf = storage_buffer(gpu, "ember-mesh-vertices", bytemuck::cast_slice(&vertices))?;
    let index_buf = storage_buffer(gpu, "ember-mesh-indices", bytemuck::cast_slice(&params.mesh.indices))?;
    let mesh_params = uniform_buffer(
        gpu,
        "ember-mesh",
        bytemuck::bytes_of(&MeshGpu {
            dims,
            dx,
            tri_count: params.mesh.triangle_count() as u32,
            offset: params.offset,
            _pad: [0; 2],
        }),
    )?;
    let mut batch = ComputeBatch::new();
    let group = bind_group(
        gpu,
        &cell_pipe,
        &[
            Bind::Tex(out.sdf),
            Bind::Buf(&mesh_params),
            Bind::Buf(&shape),
            Bind::Buf(&vertex_buf),
            Bind::Buf(&index_buf),
        ],
    )?;
    batch.dispatch(&cell_pipe, &group, cells);
    for axis in Axis::ALL {
        let face_params = uniform_buffer(
            gpu,
            "ember-mesh-faces",
            bytemuck::bytes_of(&FaceGpu { dims, dx, axis: axis_index(axis), _pad: [0; 3] }),
        )?;
        let face = out.velocity.face(axis);
        let group = bind_group(
            gpu,
            &face_pipe,
            &[Bind::Tex(face), Bind::Buf(&face_params), Bind::Buf(&shape)],
        )?;
        batch.dispatch(&face_pipe, &group, face.dims());
    }
    batch.submit(gpu)
}

#[derive(Debug, Clone)]
pub struct MeshCollider {
    params: MeshColliderParams,
}

impl Node for MeshCollider {
    fn kind(&self) -> &'static str {
        KIND
    }

    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![],
            outputs: vec![SocketType::Field, SocketType::VectorField],
        }
    }

    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let time = ctx.time();
        let pose = self.params.transform.pose(f64::from(time.frame), time.dt);
        let dx = ctx.voxel_size();
        let params = &self.params;
        produce(ctx, 1, |gpu, cache, cells, velocity| {
            fill_mesh_collider(
                gpu,
                cache,
                params,
                &pose,
                dx,
                ColliderFields { sdf: &cells[0], velocity },
            )
        })
    }
}

pub(crate) fn build(value: &serde_json::Value) -> Result<Box<dyn Node>, DocError> {
    let p: MeshColliderParams = params::parse(KIND, value)?;
    p.mesh.validate(KIND)?;
    p.transform.validate(KIND)?;
    params::finite(KIND, "offset", &[p.offset])?;
    if p.offset < 0.0 {
        return Err(params::bad(KIND, format!("offset must not be negative, got {}", p.offset)));
    }
    Ok(Box::new(MeshCollider { params: p }))
}
```

If a `collider.rs` signature differs from what this sketch assumes (e.g. `produce`'s closure parameter types, `ColliderFields::check` visibility — it is `pub(crate)`, fine inside the crate), copy `collider.rs:130-168` exactly and adapt only the body; verify against the source. Register the node in `lib.rs::register`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo nextest run -p elements-ember --test mesh_collider`
Expected: 6 passed. A large `worst error` on the box test points at sign or tie handling: debug with a single cell before changing tolerances.

- [ ] **Step 7: Prove the tests can fail**

1. In `mesh_sdf.wgsl` drop the `ndot < 0.0` sign (always `+1`): `a_box_mesh_matches...` must fail inside the box.
2. Remove the `nl <= 0.0` skip: `a_degenerate_triangle...` must fail (NaN) — note in the record if the GPU returns 0 instead of NaN, in which case assert `> 0.0`-distance to the origin cell instead and re-run.
3. Change `- mesh.offset` to `+ mesh.offset`: `offset_shrinks_the_distance` must fail.
4. In `fill_mesh_collider` pass `tri_count: 1`: the box test must fail.

- [ ] **Step 8: Commit**

```bash
just check
git add crates/elements-ember
git commit -m "Add a mesh collider"
```

---

### Task 7: Thin planks block flow (FT2)

**Files:**
- Modify: `crates/elements-ember/tests/scenes.rs` (move `collider_at` into `tests/common/mod.rs`; see Step 1)
- Modify: `crates/elements-ember/tests/common/mod.rs`
- Create: `crates/elements-ember/tests/mesh_leak.rs`

**Interfaces:**
- Consumes: `fill_mesh_collider`, `MeshColliderParams` (Task 6); `solidify`, `Uniforms`, `StepConstants`, `Solids`, `Sources`, `SolverState`, `substep`, `PressureSolve` exactly as `tests/scenes.rs::collider_scene` uses them (read `tests/scenes.rs:1-30` for its `use` lines).
- Produces: `common::solid_from_sdf(gpu, cache, pool, cells, dx, sdf: Field, velocity: StaggeredField) -> (Field /*mask*/)` and the leak test. The shared helper is generic over the producer of the SDF, so `scenes.rs::collider_at` becomes a thin wrapper.

- [ ] **Step 1: Factor the shared helper**

In `tests/common/mod.rs` add (copying the `solidify` tail of `scenes.rs::collider_at`):

```rust
/// Threshold `sdf` (< 0 is solid) into a 0/1 mask field, as the solver does.
pub fn solid_mask(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    pool: &mut FieldPool,
    cells: FieldDims,
    dx: f32,
    sdf: &Field,
) -> Field {
    use elements_ember::kernels::{StepConstants, Uniforms, solidify};
    let mask = pool.acquire(gpu, cells, FieldFormat::R32Float).unwrap();
    let u = Uniforms::new(gpu, &StepConstants::new(cells, 1.0, dx)).unwrap();
    let mut batch = elements_core::gpu::ComputeBatch::new();
    solidify(gpu, cache, &mut batch, &u, sdf, &mask).unwrap();
    batch.submit(gpu).unwrap();
    mask
}
```

Adjust the `use` paths to whatever `tests/scenes.rs` imports for `solidify`/`Uniforms`/`StepConstants` (they are not `kernels::…` necessarily; copy its lines). In `scenes.rs` make `collider_at` call `solid_mask` for its mask step so there is one copy. `just check` must pass with no behaviour change.

- [ ] **Step 2: Write the failing leak test** (`tests/mesh_leak.rs`)

Scene: 32³, 2 m domain (dx = 0.0625 m). A smoke source sits at x = 0.6 (centre 1.0, 1.0). A vertical wall plank between the source and the far side: a box mesh 0.03 m thick (thinner than half a voxel) in x, spanning y and z fully, centred at x = 1.0 — a full wall that splits the domain. The source emits density 1, temperature 2, and a velocity push toward +x (sources' `velocity` via the solver's `Sources`: take the same `Sources`/`StepConstants` the `collider_scene` test builds and add the emitter velocity fields it already builds; if `collider_scene` has no velocity push, use buoyancy only and place the wall horizontally at z = 1.0 with the source below it, so the plume rises into the wall).

Use the horizontal variant (it needs no extra plumbing): wall `Mesh::box_mesh([0.0, 0.0, -0.015], [2.0, 2.0, 0.015])` at `Transform::at([1.0, 1.0, 1.0])`, source at z = 0.3 as in `collider_scene`, 40 substeps. Measure the **mean density above the wall** (cells with z > 1.2), and compare three runs:

```rust
/// Mean density above the wall after 40 frames of a plume rising into it.
fn above_wall(offset: f32, with_wall: bool) -> f64 { /* build as collider_scene does, with the mesh collider's SDF through `solid_mask` */ }

#[test]
fn a_plank_thinner_than_a_voxel_blocks_the_plume_only_with_an_offset() {
    let open = above_wall(0.0, false);
    let leaky = above_wall(0.0, true);
    let sealed = above_wall(0.5 * 0.0625, true);
    assert!(open > 1e-3, "control: the open plume reaches z>1.2 ({open})");
    assert!(leaky > 0.2 * open, "without an offset the thin plank leaks ({leaky} vs {open})");
    assert!(sealed < 0.01 * open, "with offset = half a voxel it seals ({sealed} vs {open})");
}
```

Write `above_wall` by copying `collider_scene` (`tests/scenes.rs:409`), replacing its `ColliderParams`/`collider_at` with `fill_mesh_collider` into an `sdf` field followed by `solid_mask`, and adding a `Solids { mask, velocity }` as it does. The wall's `StaggeredField` velocity is zero (static).

- [ ] **Step 3: Run**

Run: `cargo nextest run -p elements-ember --test mesh_leak`
Expected: the three assertions decide the test. If `leaky` does **not** leak (a 0.03 m plank at dx 0.0625 has cell centres at 1.03125 and 0.96875, both 0.0156 m from the plank centre, so the plank is *just* inside them: sdf = +0.0156 − 0.015 = +0.0006 m, i.e. unsealed, which is the point), check the sign conventions by printing the SDF column through the wall; do not loosen the assertion until you have confirmed what the solid mask holds. If the sealed run still leaks, print the mask and check that `solidify` sees `sdf < 0` on both z layers; the offset must make sdf ≈ −0.031 m.

- [ ] **Step 4: Prove the test can fail**

Change `sealed`'s offset to `0.25 * 0.0625` (too small to reach the cell centres: −0.0156−… check by computing; choose the smallest value that leaves `sdf ≥ 0` at both layers, i.e. `offset = 0.0005`): the `sealed` assertion must fail. Restore. Record the output.

- [ ] **Step 5: Commit**

```bash
just check
git add crates/elements-ember/tests
git commit -m "Test that a thin plank seals only with an offset"
```

---

### Task 8: The shack as a mesh collider, cost, and docs (FT2)

**Files:**
- Modify: `crates/elements-ember/tests/mesh_collider.rs` (shack SDF test)
- Create: `docs/bench/mesh-collider.md` (cost record)
- Modify: `AGENTS.md` (document list), `.superpowers/sdd/progress.md` (ledger)

**Interfaces:**
- Consumes: `shack(&ShackParams) -> Mesh` (Task 3), `fill_mesh_collider` (Task 6).
- Produces: `docs/bench/mesh-collider.md` with measured SDF fill time at 128³ and 256×128×128 for the shack mesh.

- [ ] **Step 1: Write the failing test** (append to `tests/mesh_collider.rs`)

```rust
/// The shack is concave and open: a cell in the middle of the room is outside
/// (positive), a cell inside a wall plank is inside (negative, with the
/// offset), and the room's centre is farther from any wall than the wall
/// surface is from itself.
#[test]
fn a_concave_shack_keeps_its_interior_outside_the_solid() {
    use elements_ember::shack::{ShackParams, shack};
    let p = ShackParams {
        size: [1.2, 1.0, 1.0],
        plank_height: 0.1,
        thickness: 0.05,
        gap: 0.0,
        broken_fraction: 0.0,
        seed: 1,
    };
    let params = MeshColliderParams {
        mesh: shack(&p),
        transform: at([1.0, 1.0, 0.0], None),
        offset: 0.0,
    };
    let (sdf, _) = fill(&params, 0.0);
    let cells = FieldDims::new(N, N, N);
    // Room centre is (1.0, 1.0, 0.5): cell (16, 16, 8) is at (1.03, 1.03, 0.53).
    let inside_room = sdf[index(cells, 16, 16, 8)];
    assert!(inside_room > 0.2, "the room is open air, sdf {inside_room}");
    // The south wall plank spans y in [0.5, 0.55] (shack y = -0.5..-0.45 + 1.0):
    // cell row 8 is y = 0.53 m, inside the plank (x within ±0.6 of 1.0).
    let in_wall = sdf[index(cells, 16, 8, 8)];
    assert!(in_wall < 0.0, "the wall plank is solid, sdf {in_wall}");
    // Above the roof (z = 1.05+) is outside again.
    let above = sdf[index(cells, 16, 16, 20)];
    assert!(above > 0.0, "above the roof, sdf {above}");
}
```

This needs planks ≥ one voxel (thickness 0.05 < dx 0.0625): the in-wall cell centre at y = 0.53125 lies inside [0.5, 0.55]. If `gap: 0.0` produces coincident faces between rows, the SDF is still correct (touching boxes); note if the test shows artefacts.

- [ ] **Step 2: Run, then fix the geometry assumptions if the indices are off**

Run: `cargo nextest run -p elements-ember --test mesh_collider a_concave_shack`
Expected: PASS if the index arithmetic in the comments is right. If it fails, recompute the cell → metre mapping (`(i + 0.5) × 0.0625`) for each asserted cell and correct the *test's cell indices or expected signs*; the kernel is already proven by Task 6, so a failure here points at the test's geometry first.

- [ ] **Step 3: Prove it can fail**

Set the shack `transform` translate to `[1.0, 1.0, 0.6]` (raises the shack): the `inside_room`/`in_wall` assertions must fail. Restore.

- [ ] **Step 4: Measure the SDF fill cost**

Write a `#[test] #[ignore]` in `tests/mesh_collider.rs` (`cargo nextest run --run-ignored ignored-only`) that fills the shot's shack (`size [1.2,1.0,1.0], plank_height 0.1, thickness 0.02, gap 0.01, broken_fraction 0.2, seed 7`, about 400 triangles) at `FieldDims::new(128,128,128)` and `FieldDims::new(256,128,128)`, runs it 5 times, and prints the median wall time per fill (time `fill_mesh_collider` end to end; it submits and waits inside `batch.submit`; if it does not wait, call `gpu.wait` as other benches do). Record in `docs/bench/mesh-collider.md`: triangle count, the two dims, median and range of the times, `uptime` load before and after, adapter name, and the verdict: whether a per-frame brute-force fill is acceptable for a 240-frame bake (acceptable if under 10% of a solver step; at 256×128×128 a solver step is tens of ms to hundreds of ms per `docs/bench/results.md`). If it is not acceptable, do not optimise here: record it as risk (o) in `docs/superpowers/specs/2026-09-21-ember-solver-design.md` §6 and stop for the user's decision (options: skip the SDF when the pose did not change by caching in node state; a coarse AABB early-out).

- [ ] **Step 5: Update docs and ledger**

Add to `AGENTS.md`'s document list: the roadmap spec, the FT0 shot spec, this plan, and `docs/bench/mesh-collider.md`; and one sentence to the status paragraph: "Flamethrower vs. shack (Track A of the roadmap) is in progress: FT0–FT2 add a nozzle cone, mesh colliders and a procedural shack." Append a section to `.superpowers/sdd/progress.md` recording each task's commits and the decisions above.

- [ ] **Step 6: Commit**

```bash
just check
git add crates/elements-ember/tests/mesh_collider.rs docs AGENTS.md .superpowers/sdd/progress.md
git commit -m "Test the shack as a collider and record its SDF cost"
```

---

### Task 9: Fire divergence experiments on the existing fire scene (FT3a)

An experiment task, not a feature: the output is numbers and a recommendation. It needs no FT1/FT2 code and can run beside Tasks 4–8 (use a scratch branch or an uncommitted patch for the code experiments; commit only the results document).

**Files:**
- Create: `docs/bench/fire-divergence.md`
- Read: `.superpowers/sdd/fire-diagnosis.md`, `docs/bench/results.md` (Fire), `.superpowers/sdd/handoff-2026-09-26.md` (item 1), `justfile` (bench recipes)

**Interfaces:**
- Consumes: the fire benchmark scene and its divergence metric (`crates/elements-ember/src/metrics.rs`, `just bench`'s `fire` scene; `just bench fire 64` if the recipe accepts a scene and resolution, else read `justfile`).
- Produces: a table of RMS divergence at frame 60 per variant and resolution, and a recommendation the user decides on.

- [ ] **Step 1: Baseline.** On `main`, run the fire scene at 64³ and 128³ (`just bench fire 64`, then `128`; 256³ only if time allows and only with the machine idle) and record RMS divergence at frame 60 and frame time. Note the load (`uptime`) before and after; if load is above 4, pause background jobs and say so, because load invalidates timing (not divergence).

- [ ] **Step 2: Euler for every grid.** Find where the solver chooses the backtrace while fire burns: `grep -n -i "euler" crates/elements-ember/src/solver.rs crates/elements-ember/src/kernels/*.rs crates/elements-ember/src/kernels/shaders/*.wgsl`. On a scratch branch, make scalars (density, temperature, fuel, react) use the same Euler backtrace as velocity, run the same scene/resolutions, and record divergence. Also record peak |u| (the diagnosis measured stability this way).

- [ ] **Step 3: `flame_vorticity = 0`.** Run the baseline with `flame_vorticity` set to 0 (the fire scene's settings live in the benchmark scene definition in `crates/elements-ember/src/bench/mod.rs`; change only that value) and record divergence.

- [ ] **Step 4: `final` preset.** Run the baseline with the `final` preset (8 substeps, RK2 per `docs/bench/presets.md`) and record divergence and frame time.

- [ ] **Step 5: Write `docs/bench/fire-divergence.md`** with: the variants, the table (variant × resolution: divergence RMS at f60, frame ms, peak |u|, machine load), what the gap tracks (the variant that closes it, if any), and a recommendation for the user. No solver change is committed by this task; if a variant closes the gap without hurting fuel or flame shape, the recommendation is to scope a follow-up task, which the user decides.

- [ ] **Step 6: Commit the results only**

```bash
git add docs/bench/fire-divergence.md
git commit -m "Record which fire variant closes the divergence gap"
```

---

### Task 10: Jet-into-shack divergence and the FT3 decision (FT3b)

Depends on Tasks 5 and 8 (nozzle and shack mesh).

**Files:**
- Modify: `docs/bench/fire-divergence.md`
- Create: `examples/jet_shack.elements`

**Interfaces:**
- Consumes: `ember.emitter` with a `Cone` shape and `velocity_local` (Tasks 4–5), `ember.mesh_collider` with `shack` geometry (Tasks 3, 6), the solver's fire inputs (emitter outputs 0–4 into solver inputs 0–3 and 6; collider outputs into solver inputs 4 and 5; see `sockets()` in `solver.rs:1424`).
- Produces: a jet-into-shack document in `examples/` and the FT3 decision record.

- [ ] **Step 1: Generate the scene document.** Write a small `#[ignore]` test in `tests/mesh_collider.rs` (or a `cargo run --example` if the crate has examples; it does not, so use the test) that builds the shot's scene from Task 3's FT0 spec — shack mesh inline via `serde_json::to_value(&MeshColliderParams{..})`, the nozzle emitter (cone, `velocity [14,0,0]`, `velocity_local true`, `fuel_rate`, `temperature_rate`, `active_frames [10,150]`), solver `preset` settings copied from `bench/mod.rs`'s fire scene — and writes it to `examples/jet_shack.elements`. Commit the generated file; a human-readable diff is not expected for the inline mesh.

- [ ] **Step 2: Run it.** Bake 60 frames at `[128, 64, 64]`, `domain_size 2.0` with the CLI (`cargo run --release -p elements-cli -- --help` for the subcommand) under the baseline and each variant from Task 9 that closed or narrowed the gap. Record RMS divergence at f60, blow-up (NaN or peak |u| > 30 m/s), and whether the jet reaches the shack.

- [ ] **Step 3: Write the decision** in `docs/bench/fire-divergence.md`: the jet-into-shack numbers, the fire-scene numbers, and one of: (a) quality is good enough for the shot; (b) a named variant closes the gap and should be scoped as a solver task (name it); (c) neither helps, and the gap is accepted and recorded for FT6's verdict. Stop and ask the user to decide; do not change the solver.

- [ ] **Step 4: Commit**

```bash
git add docs/bench/fire-divergence.md examples/jet_shack.elements crates/elements-ember/tests/mesh_collider.rs
git commit -m "Measure fire divergence with a jet into the shack"
```

---

## Self-Review (spec coverage)

- **FT0** shot spec, non-cubic check, budget, procedural shack → Tasks 1, 2 (Mesh), 3.
- **FT1** cone jet emitter, velocity, noise (already present), fuel rate (already present), pulse via existing `active_frames` (one pulse per emitter; unions cover more) → Tasks 4, 5. Noise is unchanged and already tested in `tests/noise_emitter.rs`.
- **FT2** mesh → SDF and face velocity, concave, thin-walled, analytic checks, leak test, risk (m) measured → Tasks 2, 6, 7, 8. Risk (m) (preview's ×4 MGPCG under-converging around thin colliders) is measured indirectly by the leak test's pressure solve; a dedicated MGPCG-vs-GS comparison on a thin plank is **not** included and is noted in Task 8's doc update for FT6.
- **FT3** divergence experiments and the decision → Tasks 9, 10.
- Not in this plan: FT4 surface ignition, FT5 export and render template, FT6 shot assembly, Track B. Each gets its own plan.

**Known limits to watch while executing**
- Test cell-index arithmetic (Tasks 4, 7, 8) is hand-derived; if a check fails, verify the test's coordinates against `(i + 0.5) × dx` before changing kernels.
- The kernel's brute-force cost (Task 8 Step 4) may force an optimisation task before FT6; that is a decision for the user, not a silent scope change.
- `velocity_local` adds one field to `EmitterParams`; `deny_unknown_fields` means older documents without it still load (it defaults), and newer documents fail on older builds, which is acceptable.
