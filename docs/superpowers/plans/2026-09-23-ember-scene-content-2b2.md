# Ember Piece 2b-2 — Scene Content Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the Ember solver scene content: box and sphere emitters and colliders with keyframed motion, noise-modulated emission, velocity emission, wind, the collider validation scene, and the `plume_collider` / `plume_wind` benchmark scenes.

**Architecture:**
- **New nodes.** Emitters and colliders are graph nodes that output fields (umbrella E4). Both share one shape-and-transform module: the CPU evaluates keyframes into a pose, and one WGSL module computes signed distance and velocity from it.
- **Solver inputs.** The solver gains four optional inputs: an emitter velocity weight and target velocity, and a collider SDF and velocity. Each frame, a solid mask is built from the collider SDF.
- **Kernel changes.** Every kernel that handles boundaries also reads the solid mask. When no collider is connected, a `has_solids` flag in the uniform and a 1×1×1 placeholder texture stand in, so collider-free scenes run the same code paths at no measurable cost.

**Tech Stack:** Rust 2024, wgpu 30 (WGSL compute), serde, cargo-nextest; `just` recipes.

**Spec:** `docs/superpowers/specs/2026-09-23-ember-scene-content-2b2-design.md`. Read it first; §n below refers to it. **Two spec corrections are made while writing this plan and applied in Tasks 2 and 5:**
- The emitter's third output is a **velocity weight**, occupancy × `velocity_blend` in 1/s, not occupancy. A union of emitters with different blend rates cannot be expressed from occupancy alone. The solver blends with `1 − exp(−w_face · h)`.
- Scalar sampling next to a solid replaces each solid corner with **the mean of the fluid corners of the same 8** (0 if all 8 are solid). This replaces "clamp to the nearest fluid value along the axis", which has no single meaning in 3D.

## Global Constraints

- Scalar fields are `R32Float`, never `R16Float`.
- `required_features` stays `wgpu::Features::empty()`. No optional wgpu features.
- At most **4 storage textures per shader stage**. Read neighbours through `texture_3d<f32>` + `textureLoad`, and write through storage.
- `#![forbid(unsafe_code)]` in `elements-core` and `elements-ember`.
- Crate manifests use `dep.workspace = true`; versions live only in the workspace `Cargo.toml`.
- Frames are bit-identical on one machine however they are reached: in order, by scrubbing, or after cache eviction. Nothing time-dependent may live outside the state store: poses, solid masks and noise are recomputed from the document's time every frame.
- A snapshot for frame N is the state entering N. Never cache outputs.
- Documents are untrusted. Every parameter is validated at load as a `DocError`, and the daemon must not panic.
- On any error mid-step, every taken state value and every scratch field goes back to the pool before the error returns.
- All lengths are metres and rates per second, measured from the domain's minimum corner.
- Every stochastic node takes an explicit `seed: u64`.
- **Kernel loops:** in kernels that run many times per substep (the pressure sweep especially), write per-axis or per-side code out by hand; never loop over axes with dynamic vector indexing. That cost 40% per iteration on naga's Metal backend (piece 2 spec risk (j)). Loops in once-per-frame kernels, such as emitters, colliders and solidify, are fine.
- **Prove each test can fail:** apply the task's listed mutation (exactly one change), run the test, see it fail, restore, and record the real failure output in the commit body. If a listed mutation does not fail as described, find the smallest single change that does, record both, and say so.
- `just check` must pass before every commit. Never set `WGPU_BACKEND` locally.
- Check wgpu APIs against the vendored source, not docs.rs: `R=$(find ~/.cargo/registry/src -maxdepth 2 -type d -name 'wgpu-30*' | head -1)`.
- Commits have a plain imperative subject and a body explaining **why**, ending with
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01TSRpZQEY9fnKQqHLh5WHpU
  ```
  Commit with plain `git` (the repo is jj-colocated; do not run `jj`).
- Branch: `ember-scene-2b2` (already exists; the spec is committed on it).

## Conventions

- **Grids.** A face grid along axis `a` has dims `cells + 1` along `a`, and texel `i` sits at cell-unit position `i + grid_offset(a)`. Its world position in metres is that times `dx`. A cell grid has offset 0.5 on every axis.
- **Open mask.** Bit `2·axis + side` is set when that domain face is open. 2a's default is `1 << 5` (+z open).
- **Kernel uniform.** `Params` in `common.wgsl` mirrors `KernelParams` in `kernels/mod.rs` field for field, and a `const _: () = assert!(size_of::<…>() == N)` pins every GPU uniform's size.
- **CPU references.** They live in `crates/elements-ember/tests/common/mod.rs` and mirror the WGSL's operation order.
- **Shader includes.** They are concatenated with `concat!(include_str!(…), …)`. A WGSL function may only reference bindings that every kernel including its file declares. That is why `solid.wgsl` (Task 5) is a separate include, like `velocity.wgsl`.

## File map

| File | Change | Responsibility |
|---|---|---|
| `crates/elements-core/src/graph/node.rs` | modify (T1) | `EvalCtx::input_connected`, `NodeError::IncompletePair` |
| `crates/elements-ember/src/transform.rs` | create (T1) | `Shape`, `Key`, `Transform`, `Pose`, `ShapeGpu`, quaternion maths |
| `crates/elements-ember/src/kernels/shaders/shape.wgsl` | create (T1) | `Shape` uniform, `shape_sdf`, `shape_velocity` |
| `crates/elements-ember/src/shape_emitter.rs` + `shaders/emitter_*.wgsl` | create (T2, T3) | `ember.emitter`, `fill_emitter`, noise |
| `crates/elements-ember/src/collider.rs` + `shaders/collider_*.wgsl` | create (T4) | `ember.collider`, `fill_collider` |
| `crates/elements-ember/src/unions.rs` + `shaders/*_union_*.wgsl`, `shaders/weights.wgsl` | create (T2, T4) | `ember.emitter_union`, `ember.collider_union` |
| `crates/elements-ember/src/kernels/shaders/solid.wgsl`, `solidify.wgsl` | create (T5) | solid-mask helpers and the mask kernel |
| `crates/elements-ember/src/kernels/*.rs`, `shaders/*.wgsl` | modify (T5, T6) | solids in every boundary place; blend; wind |
| `crates/elements-ember/src/solver.rs` | modify (T5, T6) | six sockets, `Sources`, `Solids`, `Emission`, wind |
| `crates/elements-ember/src/bench.rs` | modify (T7) | `plume_collider`, `plume_wind` |

---

## Task 1: Core `input_connected`, shapes and keyframed transforms

Spec §2.1 and §3.1.

**Files:**
- Modify: `crates/elements-core/src/graph/node.rs`
- Create: `crates/elements-ember/src/transform.rs`, `crates/elements-ember/src/kernels/shaders/shape.wgsl`
- Modify: `crates/elements-ember/src/lib.rs` (`pub mod transform;`)
- Test: `crates/elements-core/tests/input_connected.rs` (create), `crates/elements-ember/tests/transform.rs` (create)

**Interfaces:**
- Produces (core):
  - `EvalCtx::input_connected(&self, index: u32) -> bool`
  - `NodeError::IncompletePair { node: NodeId, connected: u32, missing: u32 }`
- Produces (ember, `elements_ember::transform`):
  - `enum Shape { Sphere { radius: f32 }, Box { half_extents: [f32; 3] } }`, with `Shape::validate(&self, kind: &str) -> Result<(), DocError>`
  - `struct Rotate { axis: [f32; 3], degrees: f32 }`
  - `struct Key { frame: f32, translate: [f32; 3], rotate: Option<Rotate> }`
  - `struct Transform { keys: Vec<Key> }`, with `Transform::validate(&self, kind: &str) -> Result<(), DocError>` and `Transform::pose(&self, frame: f64, seconds_per_frame: f64) -> Pose`, plus `Transform::at(translate: [f32; 3]) -> Transform` (one key at frame 0, no rotation)
  - `struct Pose { rotation: [f64; 4] /* w, x, y, z */, translate: [f64; 3], linear: [f64; 3], angular: [f64; 3] }`, with `Pose::world_to_local(&self) -> [[f64; 3]; 3]` (row-major), `Pose::velocity_at(&self, x: [f64; 3]) -> [f64; 3]`, and `Pose::angle_about(&self, axis: [f64; 3]) -> f64` (for tests)
  - `pub(crate) struct ShapeGpu` (112 bytes, mirrors `Shape` in `shape.wgsl`), with `ShapeGpu::new(shape: &Shape, pose: &Pose) -> ShapeGpu`
  - `shape.wgsl`, which kernels include with `include_str!("kernels/shaders/shape.wgsl")` inside `concat!`. `concat!` needs literal paths, so there is no Rust constant for it.

- [ ] **Step 1: Write the failing core test**

Create `crates/elements-core/tests/input_connected.rs`:

```rust
use std::sync::{Arc, Mutex};

use elements_core::gpu::{FieldDims, FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{
    EvalCtx, Graph, Node, NodeError, SocketId, SocketSpec, SocketType, StateStore, Time, Value,
};
use elements_core::nodes::ConstantField;

/// Two field inputs, one output; records what `input_connected` said.
struct Probe(Arc<Mutex<Vec<bool>>>);

impl Node for Probe {
    fn kind(&self) -> &'static str {
        "test.probe"
    }
    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![SocketType::Field, SocketType::Field],
            outputs: vec![SocketType::Field],
        }
    }
    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        *self.0.lock().unwrap() = vec![ctx.input_connected(0), ctx.input_connected(1)];
        Ok(vec![ctx.take_input(0)?])
    }
}

#[test]
fn an_unconnected_input_is_reported_and_may_stay_unread() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut g = Graph::new();
    let source = g.add_node(Box::new(ConstantField { value: 1.0 }));
    let probe = g.add_node(Box::new(Probe(Arc::clone(&seen))));
    g.connect(
        SocketId { node: source, index: 0 },
        SocketId { node: probe, index: 0 },
    )
    .unwrap();
    g.set_output(probe);
    let gpu = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut state = StateStore::new();
    let evaluated = g
        .eval_frame(
            &gpu,
            &mut pool,
            &mut pipelines,
            &mut state,
            Time::at(1, 1, 24.0),
            FieldDims::new(4, 4, 4),
        )
        .unwrap();
    evaluated.value.release_to(&mut pool);
    assert_eq!(*seen.lock().unwrap(), vec![true, false]);
}
```

Run: `cargo nextest run -p elements-core --test input_connected`
Expected: compile error, no method `input_connected`.

- [ ] **Step 2: Implement the core additions**

In `crates/elements-core/src/graph/node.rs`, add to `impl EvalCtx<'_>`, after `input`:

```rust
    /// Whether input `index` has an edge. The graph lets an input stay
    /// unconnected until a node reads it, so a node with optional inputs asks
    /// here before reading.
    pub fn input_connected(&self, index: u32) -> bool {
        self.sources
            .get(index as usize)
            .copied()
            .flatten()
            .is_some()
    }
```

and add to `NodeError`, before `Gpu`:

```rust
    #[error("node {node:?} input {connected} is connected but input {missing}, which it needs, is not")]
    IncompletePair {
        node: NodeId,
        connected: u32,
        missing: u32,
    },
```

Run `cargo build --workspace --all-targets` to confirm that no exhaustive `match` on `NodeError` breaks. The daemon maps non-GPU errors through a wildcard arm. Then run the test again: 1 passed. Mutation: return `true` always. Expected: FAIL with `[true, true]`. Record it, restore.

- [ ] **Step 3: Write the failing transform tests**

Create `crates/elements-ember/tests/transform.rs`:

```rust
use elements_ember::transform::{Key, Rotate, Shape, Transform};

const SPF: f64 = 1.0 / 24.0;

fn key(frame: f32, translate: [f32; 3], rotate: Option<([f32; 3], f32)>) -> Key {
    Key {
        frame,
        translate,
        rotate: rotate.map(|(axis, degrees)| Rotate { axis, degrees }),
    }
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol, "{what}: {a} vs {b}");
}

/// Spec §2.1: translation is linear between keys, and velocity is the
/// segment's slope in m/s.
#[test]
fn translation_interpolates_linearly_and_velocity_is_its_slope() {
    let t = Transform {
        keys: vec![key(0.0, [0.0; 3], None), key(10.0, [1.0, 2.0, 0.0], None)],
    };
    let p = t.pose(5.0, SPF);
    close(p.translate[0], 0.5, 1e-12, "x");
    close(p.translate[1], 1.0, 1e-12, "y");
    let seconds = 10.0 * SPF;
    close(p.linear[0], 1.0 / seconds, 1e-9, "vx");
    close(p.linear[1], 2.0 / seconds, 1e-9, "vy");
}

/// Rotation is a slerp: a quarter of the way through a 170° turn is 42.5°.
/// A normalised linear blend of quaternions lands measurably elsewhere.
#[test]
fn rotation_interpolates_by_slerp_with_constant_angular_velocity() {
    let t = Transform {
        keys: vec![
            key(0.0, [0.0; 3], None),
            key(8.0, [0.0; 3], Some(([0.0, 0.0, 1.0], 170.0))),
        ],
    };
    let p = t.pose(2.0, SPF);
    close(p.angle_about([0.0, 0.0, 1.0]), 42.5f64.to_radians(), 1e-6, "angle");
    let omega = 170f64.to_radians() / (8.0 * SPF);
    close(p.angular[2], omega, 1e-6, "ωz");
    close(p.angular[0], 0.0, 1e-9, "ωx");
}

#[test]
fn a_transform_holds_before_its_first_key_and_after_its_last() {
    let t = Transform {
        keys: vec![key(5.0, [1.0, 0.0, 0.0], None), key(9.0, [3.0, 0.0, 0.0], None)],
    };
    for (frame, x) in [(0.0, 1.0), (5.0, 1.0), (9.0, 3.0), (30.0, 3.0)] {
        let p = t.pose(frame, SPF);
        close(p.translate[0], x, 1e-12, &format!("x at {frame}"));
        if frame < 5.0 || frame >= 9.0 {
            assert_eq!(p.linear, [0.0; 3], "still at {frame}");
        }
    }
}

/// v(x) = linear + ω × (x − translate).
#[test]
fn material_velocity_adds_the_spin_about_the_origin() {
    let t = Transform {
        keys: vec![
            key(0.0, [1.0, 0.0, 0.0], None),
            key(24.0, [1.0, 0.0, 0.0], Some(([0.0, 0.0, 1.0], 90.0))),
        ],
    };
    let p = t.pose(12.0, SPF);
    let omega = std::f64::consts::FRAC_PI_2 / 1.0; // 90° over 24 frames = 1 s
    let v = p.velocity_at([1.0, 1.0, 0.0]);
    close(v[0], -omega, 1e-9, "vx");
    close(v[1], 0.0, 1e-9, "vy");
}

fn rejected_transform(keys: Vec<Key>) -> bool {
    Transform { keys }.validate("test").is_err()
}

#[test]
fn bad_shapes_and_transforms_are_rejected() {
    assert!(rejected_transform(vec![]));
    assert!(rejected_transform(vec![key(3.0, [0.0; 3], None), key(3.0, [0.0; 3], None)]));
    assert!(rejected_transform(vec![key(0.0, [f32::NAN, 0.0, 0.0], None)]));
    assert!(rejected_transform(vec![key(0.0, [0.0; 3], Some(([0.0; 3], 10.0)))]));
    assert!(!rejected_transform(vec![key(0.0, [0.0; 3], Some(([0.0, 1.0, 0.0], 10.0)))]));
    assert!(Shape::Sphere { radius: 0.0 }.validate("test").is_err());
    assert!(Shape::Box { half_extents: [0.1, -0.1, 0.1] }.validate("test").is_err());
    assert!(Shape::Box { half_extents: [0.1, 0.1, 0.1] }.validate("test").is_ok());
}

#[test]
fn shapes_parse_from_documents() {
    let s: Shape = serde_json::from_value(serde_json::json!({ "box": { "half_extents": [1.0, 2.0, 3.0] } })).unwrap();
    assert_eq!(s, Shape::Box { half_extents: [1.0, 2.0, 3.0] });
    let t: Transform = serde_json::from_value(serde_json::json!({
        "keys": [{ "frame": 1, "translate": [1, 0, 0], "rotate": { "axis": [0, 0, 1], "degrees": 45 } }]
    }))
    .unwrap();
    assert_eq!(t.keys.len(), 1);
    assert!(serde_json::from_value::<Shape>(serde_json::json!({ "cone": {} })).is_err());
}
```

Run: `cargo nextest run -p elements-ember --test transform`
Expected: compile error, no module `transform`.

- [ ] **Step 4: Write `transform.rs`**

Create `crates/elements-ember/src/transform.rs`:

```rust
//! Shapes and keyframed transforms, shared by emitters and colliders
//! (2b-2 spec §2.1). The CPU evaluates the keys into a `Pose`; the GPU only
//! reads the resulting matrix and velocities.

use elements_core::graph::DocError;
use serde::{Deserialize, Serialize};

use crate::params;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", deny_unknown_fields)]
pub enum Shape {
    Sphere { radius: f32 },
    Box { half_extents: [f32; 3] },
}

impl Shape {
    pub fn validate(&self, kind: &str) -> Result<(), DocError> {
        match self {
            Self::Sphere { radius } => {
                params::finite(kind, "radius", &[*radius])?;
                if *radius <= 0.0 {
                    return Err(params::bad(kind, format!("radius must be positive, got {radius}")));
                }
            }
            Self::Box { half_extents } => {
                params::finite(kind, "half_extents", half_extents)?;
                if half_extents.iter().any(|&e| e <= 0.0) {
                    return Err(params::bad(
                        kind,
                        format!("half_extents must be positive, got {half_extents:?}"),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rotate {
    pub axis: [f32; 3],
    pub degrees: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Key {
    pub frame: f32,
    #[serde(default)]
    pub translate: [f32; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotate: Option<Rotate>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Transform {
    pub keys: Vec<Key>,
}

/// Where an object is at one moment and how it moves, in world space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    /// Local-to-world rotation as a unit quaternion, [w, x, y, z].
    pub rotation: [f64; 4],
    pub translate: [f64; 3],
    /// dT/dt, m/s.
    pub linear: [f64; 3],
    /// ω, rad/s.
    pub angular: [f64; 3],
}

type Quat = [f64; 4];

fn q_mul(a: Quat, b: Quat) -> Quat {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn q_conj(q: Quat) -> Quat {
    [q[0], -q[1], -q[2], -q[3]]
}

fn q_dot(a: Quat, b: Quat) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

fn q_axis_angle(r: Option<Rotate>) -> Quat {
    let Some(r) = r else {
        return [1.0, 0.0, 0.0, 0.0];
    };
    let a = r.axis.map(f64::from);
    let len = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let half = f64::from(r.degrees).to_radians() * 0.5;
    let s = half.sin() / len;
    [half.cos(), a[0] * s, a[1] * s, a[2] * s]
}

/// Spherical interpolation along the shorter arc.
fn q_slerp(a: Quat, b: Quat, t: f64) -> Quat {
    let mut b = b;
    let mut d = q_dot(a, b);
    if d < 0.0 {
        b = b.map(|x| -x);
        d = -d;
    }
    if d > 1.0 - 1e-12 {
        return a;
    }
    let theta = d.clamp(-1.0, 1.0).acos();
    let (s0, s1) = (((1.0 - t) * theta).sin(), (t * theta).sin());
    let inv = 1.0 / theta.sin();
    std::array::from_fn(|i| (a[i] * s0 + b[i] * s1) * inv)
}

/// The rotation taking `a` to `b`, as an axis times an angle (world frame),
/// along the shorter arc.
fn q_rotation_vector(a: Quat, b: Quat) -> [f64; 3] {
    let mut r = q_mul(b, q_conj(a));
    if r[0] < 0.0 {
        r = r.map(|x| -x);
    }
    let s = (r[1] * r[1] + r[2] * r[2] + r[3] * r[3]).sqrt();
    if s < 1e-15 {
        return [0.0; 3];
    }
    let angle = 2.0 * s.atan2(r[0]);
    [r[1] / s * angle, r[2] / s * angle, r[3] / s * angle]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

impl Transform {
    /// One key at frame 0: a static object at `translate`.
    pub fn at(translate: [f32; 3]) -> Self {
        Self {
            keys: vec![Key {
                frame: 0.0,
                translate,
                rotate: None,
            }],
        }
    }

    pub fn validate(&self, kind: &str) -> Result<(), DocError> {
        if self.keys.is_empty() {
            return Err(params::bad(kind, "transform needs at least one key"));
        }
        for key in &self.keys {
            params::finite(kind, "key frame", &[key.frame])?;
            params::finite(kind, "translate", &key.translate)?;
            if let Some(r) = key.rotate {
                params::finite(kind, "rotate", &[r.axis[0], r.axis[1], r.axis[2], r.degrees])?;
                let len2 = r.axis.iter().map(|&a| f64::from(a) * f64::from(a)).sum::<f64>();
                if len2 < 1e-24 {
                    return Err(params::bad(kind, "rotate axis must be nonzero"));
                }
            }
        }
        if self.keys.windows(2).any(|w| w[1].frame <= w[0].frame) {
            return Err(params::bad(kind, "key frames must strictly increase"));
        }
        Ok(())
    }

    /// The pose at `frame` (it may be fractional). Velocities are per second,
    /// with `seconds_per_frame` converting the key spacing. Before the first
    /// key and from the last key on, the transform holds with zero velocity.
    /// Validate first; this assumes at least one key in increasing order.
    pub fn pose(&self, frame: f64, seconds_per_frame: f64) -> Pose {
        let hold = |k: &Key| Pose {
            rotation: q_axis_angle(k.rotate),
            translate: k.translate.map(f64::from),
            linear: [0.0; 3],
            angular: [0.0; 3],
        };
        let first = &self.keys[0];
        let last = &self.keys[self.keys.len() - 1];
        if frame < f64::from(first.frame) || self.keys.len() == 1 {
            return hold(first);
        }
        if frame >= f64::from(last.frame) {
            return hold(last);
        }
        let i = self
            .keys
            .windows(2)
            .position(|w| frame >= f64::from(w[0].frame) && frame < f64::from(w[1].frame))
            .expect("frame lies strictly inside the keyed range");
        let (k0, k1) = (&self.keys[i], &self.keys[i + 1]);
        let (f0, f1) = (f64::from(k0.frame), f64::from(k1.frame));
        let t = (frame - f0) / (f1 - f0);
        let seconds = (f1 - f0) * seconds_per_frame;
        let (t0, t1) = (k0.translate.map(f64::from), k1.translate.map(f64::from));
        let (q0, q1) = (q_axis_angle(k0.rotate), q_axis_angle(k1.rotate));
        Pose {
            rotation: q_slerp(q0, q1, t),
            translate: std::array::from_fn(|a| t0[a] + (t1[a] - t0[a]) * t),
            linear: std::array::from_fn(|a| (t1[a] - t0[a]) / seconds),
            angular: q_rotation_vector(q0, q1).map(|w| w / seconds),
        }
    }
}

impl Pose {
    /// World-to-local rotation, row-major: the transpose of local-to-world.
    pub fn world_to_local(&self) -> [[f64; 3]; 3] {
        let [w, x, y, z] = self.rotation;
        // Local-to-world matrix of a unit quaternion, then transposed.
        let m = [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ];
        std::array::from_fn(|r| std::array::from_fn(|c| m[c][r]))
    }

    /// Velocity of the object's material at world point `x`, m/s.
    pub fn velocity_at(&self, x: [f64; 3]) -> [f64; 3] {
        let r = std::array::from_fn(|a| x[a] - self.translate[a]);
        let spin = cross(self.angular, r);
        std::array::from_fn(|a| self.linear[a] + spin[a])
    }

    /// The rotation's angle about `axis` (unit length), for tests.
    pub fn angle_about(&self, axis: [f64; 3]) -> f64 {
        let [w, x, y, z] = self.rotation;
        let along = x * axis[0] + y * axis[1] + z * axis[2];
        2.0 * along.atan2(w)
    }
}

/// Matches `Shape` in `shape.wgsl`: a `mat3x3<f32>` (three 16-byte columns),
/// then four vec3 + scalar rows. 112 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ShapeGpu {
    world_to_local: [[f32; 4]; 3],
    origin: [f32; 3],
    kind: u32,
    extents: [f32; 3],
    _pad0: u32,
    linear: [f32; 3],
    _pad1: u32,
    angular: [f32; 3],
    _pad2: u32,
}

const _: () = assert!(std::mem::size_of::<ShapeGpu>() == 112);

impl ShapeGpu {
    pub(crate) fn new(shape: &Shape, pose: &Pose) -> Self {
        let m = pose.world_to_local();
        let (kind, extents) = match *shape {
            Shape::Sphere { radius } => (0, [radius, 0.0, 0.0]),
            Shape::Box { half_extents } => (1, half_extents),
        };
        Self {
            // WGSL matrices are column-major: column c holds row r's entry c.
            world_to_local: std::array::from_fn(|c| {
                [m[0][c] as f32, m[1][c] as f32, m[2][c] as f32, 0.0]
            }),
            origin: pose.translate.map(|v| v as f32),
            kind,
            extents,
            _pad0: 0,
            linear: pose.linear.map(|v| v as f32),
            _pad1: 0,
            angular: pose.angular.map(|v| v as f32),
            _pad2: 0,
        }
    }
}
```

Create `crates/elements-ember/src/kernels/shaders/shape.wgsl`:

```wgsl
// Shapes for emitters and colliders (2b-2 spec §2.1). Concatenated in front
// of a kernel that declares `var<uniform> shape: Shape`. The CPU evaluates the
// keyframes; this only reads the resulting pose.

struct Shape {
    world_to_local: mat3x3<f32>,
    origin: vec3<f32>,   // the object's origin in world space, metres
    kind: u32,           // 0 sphere, 1 box
    extents: vec3<f32>,  // sphere: radius in .x; box: half extents
    _pad0: u32,
    linear: vec3<f32>,   // dT/dt, m/s
    _pad1: u32,
    angular: vec3<f32>,  // ω, rad/s
    _pad2: u32,
};

// Signed distance from world point `x` (metres) to the surface; negative inside.
fn shape_sdf(x: vec3<f32>) -> f32 {
    let p = shape.world_to_local * (x - shape.origin);
    if (shape.kind == 0u) {
        return length(p) - shape.extents.x;
    }
    // Exact box distance: outside, the distance to the nearest point; inside,
    // minus the distance to the nearest face.
    let q = abs(p) - shape.extents;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// Velocity of the shape's material at world point `x`, m/s.
fn shape_velocity(x: vec3<f32>) -> vec3<f32> {
    return shape.linear + cross(shape.angular, x - shape.origin);
}
```

Add `pub mod transform;` to `crates/elements-ember/src/lib.rs`. `ShapeGpu` and `ShapeGpu::new` are crate-private and unused until Task 2. Put `#[allow(dead_code)] // used from Task 2` on both so clippy's `-D warnings` passes. Task 2 removes the two allows.

- [ ] **Step 5: Run and prove**

Run: `cargo nextest run -p elements-ember --test transform`
Expected: 6 passed.

| Test | Mutation | Expected |
|---|---|---|
| `rotation_interpolates_by_slerp_with_constant_angular_velocity` | in `pose`, replace `q_slerp(q0, q1, t)` with the normalised lerp `{ let q: Quat = std::array::from_fn(\|i\| q0[i] + (q1[i] - q0[i]) * t); let n = q_dot(q, q).sqrt(); q.map(\|x\| x / n) }` | FAIL: angle not 42.5° |
| `translation_interpolates_linearly_and_velocity_is_its_slope` | in `pose`, `linear` divides by `(f1 - f0)` rather than `seconds` | FAIL: vx |
| `a_transform_holds_before_its_first_key_and_after_its_last` | change `frame >= f64::from(last.frame)` to `>` | FAIL at 9.0 (the `expect` panics or velocity is nonzero) |
| `material_velocity_adds_the_spin_about_the_origin` | `velocity_at` returns `self.linear` | FAIL: vx |
| `bad_shapes_and_transforms_are_rejected` | delete the strictly-increasing check | FAIL |

- [ ] **Step 6: Run the gate and commit**

Run: `just check`
Expected: PASS.

```bash
git add crates/elements-core crates/elements-ember
git commit   # subject: "Add keyframed shapes, and let nodes ask whether an input is connected"
```

The body explains why:
- Emitters and colliders share one shape and pose model, evaluated on the CPU so the GPU never interpolates.
- The solver's new inputs are optional, and a node needs a way to ask before reading them.

Include the mutation outputs.

---
## Task 2: `ember.emitter` and `ember.emitter_union`

Spec §2.2 and §2.4, with the first correction noted at the top: output 3 is a velocity weight.

**Files:**
- Create: `crates/elements-ember/src/shape_emitter.rs`, `crates/elements-ember/src/unions.rs`, `crates/elements-ember/src/node_util.rs`
- Create: `crates/elements-ember/src/kernels/shaders/emitter_common.wgsl`, `emitter_cells.wgsl`, `emitter_faces.wgsl`, `weights.wgsl`, `emitter_union_cells.wgsl`, `emitter_union_faces.wgsl`
- Modify: `crates/elements-ember/src/lib.rs` (modules, registration), `crates/elements-ember/src/kernels/mod.rs` (`uniform_buffer`), `crates/elements-ember/src/transform.rs` (remove the Task 1 `dead_code` allows)
- Modify: `docs/superpowers/specs/2026-09-23-ember-scene-content-2b2-design.md` (output 3 and the blend formula)
- Test: `crates/elements-ember/tests/shape_emitter.rs` (create)

**Interfaces:**
- Consumes: `Shape`, `Transform`, `Pose`, `ShapeGpu`, and `shape.wgsl` (Task 1).
- Produces:
  - `shape_emitter::KIND = "ember.emitter"`
  - `EmitterParams { shape, transform, density_rate, temperature_rate, velocity: [f32; 3], velocity_blend }`, with `EmitterParams::new(shape: Shape, transform: Transform) -> Self` (every rate 0, velocity 0). Task 3 adds `noise`.
  - `EmitterFields<'a> { density: &'a Field, temperature: &'a Field, weight: &'a Field, velocity: &'a StaggeredField }`
  - `fill_emitter(gpu, cache, params: &EmitterParams, pose: &Pose, seconds: f64, dx: f32, out: EmitterFields<'_>) -> Result<(), GpuError>`
  - `unions::EMITTER_UNION_KIND = "ember.emitter_union"`, and `unions::union_emitters(gpu, cache, a: EmitterFields<'_>, b: EmitterFields<'_>, out: EmitterFields<'_>) -> Result<(), GpuError>`
  - `node_util::take_inputs(ctx: &mut EvalCtx<'_>, n: u32) -> Result<Vec<Value>, NodeError>` and `node_util::acquire_cells(ctx: &mut EvalCtx<'_>, n: usize) -> Result<Vec<Field>, NodeError>`, each releasing everything it already holds on failure
  - `kernels::uniform_buffer(gpu: &GpuContext, label: &str, bytes: &[u8]) -> Result<wgpu::Buffer, GpuError>` (pub(crate))
  - `weights.wgsl`: `fn face_weight(w: texture_3d<f32>, axis: u32, p: vec3<i32>, dims: vec3<u32>) -> f32`. The solver reuses it in Task 6.
- Node sockets:
  - `ember.emitter`: outputs `[Field density rate, Field temperature rate, Field velocity weight, VectorField target velocity]`.
  - `ember.emitter_union`: inputs are two of those groups, eight in all; outputs are one group.

- [ ] **Step 1: Write the failing tests**

Create `crates/elements-ember/tests/shape_emitter.rs`:

```rust
mod common;

use common::*;
use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, GpuContext, PipelineCache};
use elements_core::graph::DocError;
use elements_ember::shape_emitter::{EmitterFields, EmitterParams, fill_emitter};
use elements_ember::transform::{Key, Rotate, Shape, Transform};
use elements_ember::unions::union_emitters;

const SPF: f64 = 1.0 / 24.0;

/// Run one emitter at `frame` in a 2 m domain of `cells`: the three cell
/// outputs (density, temperature, weight) and the three velocity faces.
fn run(gpu: &GpuContext, cells: FieldDims, params: &EmitterParams, frame: f64) -> ([Vec<f32>; 3], [Vec<f32>; 3]) {
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let dx = 2.0 / cells.x.max(cells.y).max(cells.z) as f32;
    let f: [_; 3] = std::array::from_fn(|_| pool.acquire(gpu, cells, FieldFormat::R32Float).unwrap());
    let v = pool.acquire_staggered_uninit(gpu, cells).unwrap();
    let pose = params.transform.pose(frame, SPF);
    fill_emitter(
        gpu,
        &mut cache,
        params,
        &pose,
        frame * SPF,
        dx,
        EmitterFields { density: &f[0], temperature: &f[1], weight: &f[2], velocity: &v },
    )
    .unwrap();
    (f.map(|x| x.read_back(gpu).unwrap()), read_staggered(gpu, &v))
}

fn keyed(translate: [f32; 3], rotate: Option<Rotate>) -> Transform {
    Transform { keys: vec![Key { frame: 0.0, translate, rotate }] }
}

/// Spec §2.2: occupancy has the one-voxel smoothed edge, so the emitted total
/// matches the shape's volume at any resolution, and rotation does not change
/// it.
#[test]
fn a_box_emits_its_volume_at_any_resolution_and_orientation() {
    let gpu = gpu();
    let volume = 0.6 * 0.4 * 0.5;
    for n in [32u32, 64] {
        let cells = FieldDims::new(n, n, n);
        let dx = 2.0 / f64::from(n);
        for rotate in [None, Some(Rotate { axis: [1.0, 1.0, 0.0], degrees: 30.0 })] {
            let p = EmitterParams {
                density_rate: 1.0,
                ..EmitterParams::new(
                    Shape::Box { half_extents: [0.3, 0.2, 0.25] },
                    keyed([1.0, 1.0, 1.0], rotate),
                )
            };
            let ([d, _, _], _) = run(&gpu, cells, &p, 0.0);
            let total = d.iter().map(|&v| f64::from(v)).sum::<f64>() * dx.powi(3);
            assert!((total - volume).abs() <= 0.05 * volume, "{n}³ {rotate:?}: {total} vs {volume}");
        }
    }
}

/// A long thin box turned 90° about z points along y: the point 0.4 m along
/// +y from its centre is inside, and the point 0.4 m along +x is not.
#[test]
fn rotation_turns_the_box() {
    let gpu = gpu();
    let cells = FieldDims::new(32, 32, 32);
    let p = EmitterParams {
        density_rate: 1.0,
        ..EmitterParams::new(
            Shape::Box { half_extents: [0.5, 0.1, 0.1] },
            keyed([1.0, 1.0, 1.0], Some(Rotate { axis: [0.0, 0.0, 1.0], degrees: 90.0 })),
        )
    };
    let ([d, _, _], _) = run(&gpu, cells, &p, 0.0);
    // dx = 1/16 m: cell 22 is centred at 1.40625 m, cell 16 at 1.03125 m.
    assert_eq!(d[index(cells, 16, 22, 16)], 1.0, "along +y");
    assert_eq!(d[index(cells, 22, 16, 16)], 0.0, "along +x");
}

/// Spec §2.2: the target velocity on every face is the emitter's `velocity`
/// plus its own motion there, and the weight is occupancy × blend.
#[test]
fn target_velocity_carries_the_emitters_motion_and_weight_is_occupancy_times_blend() {
    let gpu = gpu();
    let cells = FieldDims::new(12, 10, 8);
    let dx = 2.0 / 12.0;
    let transform = Transform {
        keys: vec![
            Key { frame: 0.0, translate: [0.8, 0.8, 0.6], rotate: None },
            Key {
                frame: 24.0,
                translate: [1.2, 0.8, 0.7],
                rotate: Some(Rotate { axis: [0.0, 0.0, 1.0], degrees: 60.0 }),
            },
        ],
    };
    let p = EmitterParams {
        velocity: [0.5, -1.0, 0.25],
        velocity_blend: 2.0,
        ..EmitterParams::new(Shape::Sphere { radius: 0.3 }, transform.clone())
    };
    let frame = 6.0;
    let ([_, _, w], faces) = run(&gpu, cells, &p, frame);
    let pose = transform.pose(frame, SPF);
    for a in 0..3 {
        let d = face_dims(cells, a);
        let off = face_offset(a);
        for k in 0..d.z {
            for j in 0..d.y {
                for i in 0..d.x {
                    let x = [i as f32 + off[0], j as f32 + off[1], k as f32 + off[2]]
                        .map(|c| f64::from(c) * f64::from(dx));
                    let want = f64::from(p.velocity[a]) + pose.velocity_at(x)[a];
                    let got = f64::from(faces[a][index(d, i, j, k)]);
                    assert!((got - want).abs() <= 1e-5, "face {a} {:?}: {got} vs {want}", [i, j, k]);
                }
            }
        }
    }
    // The sphere's centre cell is fully occupied: weight = 1 × blend.
    let c = pose.translate.map(|v| (v / f64::from(dx)) as u32);
    assert_eq!(w[index(cells, c[0], c[1], c[2])], 2.0);
    assert_eq!(w[index(cells, 0, 0, 0)], 0.0);
}

/// Mirrors `face_weight` in `weights.wgsl`.
fn face_weight(w: &[f32], cells: FieldDims, axis: usize, p: [u32; 3]) -> f32 {
    let n = [cells.x as i32, cells.y as i32, cells.z as i32];
    let at = |q: [i32; 3]| {
        let c: [u32; 3] = std::array::from_fn(|a| q[a].clamp(0, n[a] - 1) as u32);
        w[index(cells, c[0], c[1], c[2])]
    };
    let mut below = p.map(|v| v as i32);
    below[axis] -= 1;
    at(below).max(at(p.map(|v| v as i32)))
}

/// Spec §2.4: rates add, the weight is the maximum, and the target velocity
/// is weighted by each emitter's face weight.
#[test]
fn a_union_adds_rates_takes_the_max_weight_and_weights_velocity() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(12, 10, 8);
    let dx = 2.0 / 12.0;
    let a = EmitterParams {
        density_rate: 1.0,
        temperature_rate: 1.0,
        velocity: [1.0, 0.0, 0.0],
        velocity_blend: 1.0,
        ..EmitterParams::new(Shape::Box { half_extents: [0.3, 0.3, 0.3] }, keyed([0.8, 0.8, 0.6], None))
    };
    let b = EmitterParams {
        density_rate: 2.0,
        temperature_rate: 0.5,
        velocity: [0.0, 0.0, 2.0],
        velocity_blend: 3.0,
        ..EmitterParams::new(Shape::Sphere { radius: 0.35 }, keyed([1.1, 0.8, 0.7], None))
    };
    let make = |pool: &mut FieldPool| {
        let f: [_; 3] = std::array::from_fn(|_| pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap());
        (f, pool.acquire_staggered_uninit(&gpu, cells).unwrap())
    };
    let (fa, va) = make(&mut pool);
    let (fb, vb) = make(&mut pool);
    let (fo, vo) = make(&mut pool);
    let fields = |f: &[elements_core::gpu::Field; 3], v| EmitterFields {
        density: &f[0],
        temperature: &f[1],
        weight: &f[2],
        velocity: v,
    };
    for (p, f, v) in [(&a, &fa, &va), (&b, &fb, &vb)] {
        let pose = p.transform.pose(0.0, SPF);
        fill_emitter(&gpu, &mut cache, p, &pose, 0.0, dx, fields(f, v)).unwrap();
    }
    union_emitters(&gpu, &mut cache, fields(&fa, &va), fields(&fb, &vb), fields(&fo, &vo)).unwrap();

    let read = |f: &[elements_core::gpu::Field; 3]| f.each_ref().map(|x| x.read_back(&gpu).unwrap());
    let (ca, cb, co) = (read(&fa), read(&fb), read(&fo));
    for n in 0..cells.voxel_count() {
        assert_eq!(co[0][n], ca[0][n] + cb[0][n], "density {n}");
        assert_eq!(co[1][n], ca[1][n] + cb[1][n], "temperature {n}");
        assert_eq!(co[2][n], ca[2][n].max(cb[2][n]), "weight {n}");
    }
    let (ua, ub, uo) = (read_staggered(&gpu, &va), read_staggered(&gpu, &vb), read_staggered(&gpu, &vo));
    for ax in 0..3 {
        let d = face_dims(cells, ax);
        for k in 0..d.z {
            for j in 0..d.y {
                for i in 0..d.x {
                    let at = index(d, i, j, k);
                    let (wa, wb) = (face_weight(&ca[2], cells, ax, [i, j, k]), face_weight(&cb[2], cells, ax, [i, j, k]));
                    let want = (wa * ua[ax][at] + wb * ub[ax][at]) / (wa + wb).max(1e-12);
                    assert!((uo[ax][at] - want).abs() <= 1e-6, "face {ax} {:?}: {} vs {want}", [i, j, k], uo[ax][at]);
                }
            }
        }
    }
}

fn rejected(params: serde_json::Value) -> bool {
    matches!(
        elements_ember::registry().build(elements_ember::shape_emitter::KIND, &params),
        Err(DocError::BadParams { .. })
    )
}

#[test]
fn bad_emitter_parameters_are_rejected() {
    let ok = serde_json::json!({ "shape": { "sphere": { "radius": 0.2 } }, "transform": { "keys": [{ "frame": 0 }] } });
    assert!(!rejected(ok.clone()));
    let with = |k: &str, v: serde_json::Value| {
        let mut o = ok.clone();
        o[k] = v;
        o
    };
    assert!(rejected(with("velocity_blend", serde_json::json!(-1.0))));
    assert!(rejected(with("density_rate", serde_json::json!(1e39))));
    assert!(rejected(with("transform", serde_json::json!({ "keys": [] }))));
    assert!(rejected(with("shape", serde_json::json!({ "box": { "half_extents": [0.1, 0.0, 0.1] } }))));
    assert!(rejected(with("colour", serde_json::json!(1.0))));
}
```

Run: `cargo nextest run -p elements-ember --test shape_emitter`
Expected: compile error, no module `shape_emitter`.

- [ ] **Step 2: Shared node helpers and the uniform helper**

Create `crates/elements-ember/src/node_util.rs`:

```rust
//! Small helpers for nodes that take several inputs or acquire several fields.

use elements_core::gpu::{Field, FieldFormat};
use elements_core::graph::{EvalCtx, NodeError, Value};

/// Take inputs `0..n` in order. On failure, every input already taken is released.
pub(crate) fn take_inputs(ctx: &mut EvalCtx<'_>, n: u32) -> Result<Vec<Value>, NodeError> {
    let mut taken = Vec::with_capacity(n as usize);
    for i in 0..n {
        match ctx.take_input(i) {
            Ok(value) => taken.push(value),
            Err(e) => {
                for value in taken {
                    ctx.release(value);
                }
                return Err(e);
            }
        }
    }
    Ok(taken)
}

/// Acquire `n` uninitialised `R32Float` fields at the domain's dims. On
/// failure, every field already acquired is released.
pub(crate) fn acquire_cells(ctx: &mut EvalCtx<'_>, n: usize) -> Result<Vec<Field>, NodeError> {
    let mut fields = Vec::with_capacity(n);
    for _ in 0..n {
        match ctx.acquire_uninit(FieldFormat::R32Float) {
            Ok(field) => fields.push(field),
            Err(e) => {
                for field in fields {
                    ctx.release(Value::Field(field));
                }
                return Err(e);
            }
        }
    }
    Ok(fields)
}
```

In `crates/elements-ember/src/kernels/mod.rs`, add:

```rust
/// A uniform buffer holding `bytes`, created inside an error scope.
pub(crate) fn uniform_buffer(gpu: &GpuContext, label: &str, bytes: &[u8]) -> Result<wgpu::Buffer, GpuError> {
    gpu.scoped(|| {
        gpu.device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            })
    })
}
```

Also make `Bind`, `bind_group`, `expect_dims` and `axis_index` reachable from sibling modules. They are already `pub(crate)`, but `mod.rs` must be declared as `pub mod kernels` in lib.rs, which it already is.

- [ ] **Step 3: The emitter shaders**

Create `crates/elements-ember/src/kernels/shaders/emitter_common.wgsl`:

```wgsl
// Shared by the emitter's cell and face passes (2b-2 spec §2.2). 80 bytes;
// mirrors `EmitterGpu` in shape_emitter.rs.

struct Emitter {
    dims: vec3<u32>,
    dx: f32,
    velocity: vec3<f32>,   // target velocity, m/s, world space
    axis: u32,             // the face pass's axis (0, 1, 2)
    density_rate: f32,
    temperature_rate: f32,
    velocity_blend: f32,   // 1/s
    noise_on: u32,
    noise_scale: f32,      // metres
    noise_amplitude: f32,
    noise_w: f32,          // evolution × seconds: noise's fourth coordinate
    seed_lo: u32,
    seed_hi: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
```

Create `crates/elements-ember/src/kernels/shaders/emitter_cells.wgsl`:

```wgsl
// ember.emitter, cell pass: density and temperature rates, and the velocity
// weight, occupancy × velocity_blend in 1/s (2b-2 spec §2.2).

@group(0) @binding(0) var density: texture_storage_3d<r32float, write>;
@group(0) @binding(1) var temperature: texture_storage_3d<r32float, write>;
@group(0) @binding(2) var weight: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var<uniform> emitter: Emitter;
@group(0) @binding(4) var<uniform> shape: Shape;

// The emission factor from noise at world point `x`. Task 3 fills this in.
fn noise_factor(x: vec3<f32>) -> f32 {
    return 1.0;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= emitter.dims)) {
        return;
    }
    let x = (vec3<f32>(gid) + vec3<f32>(0.5)) * emitter.dx;
    // A one-voxel linear ramp centred on the surface, as ember.sphere_emitter
    // uses, so the emitted total matches the volume at any resolution.
    let occupancy = clamp(0.5 - shape_sdf(x) / emitter.dx, 0.0, 1.0);
    let rate = occupancy * noise_factor(x);
    let p = vec3<i32>(gid);
    textureStore(density, p, vec4<f32>(rate * emitter.density_rate, 0.0, 0.0, 0.0));
    textureStore(temperature, p, vec4<f32>(rate * emitter.temperature_rate, 0.0, 0.0, 0.0));
    textureStore(weight, p, vec4<f32>(occupancy * emitter.velocity_blend, 0.0, 0.0, 0.0));
}
```

Create `crates/elements-ember/src/kernels/shaders/emitter_faces.wgsl`:

```wgsl
// ember.emitter, face pass: the target velocity on one axis's faces, the
// emitter's `velocity` plus its own motion there (2b-2 spec §2.2).

@group(0) @binding(0) var face: texture_storage_3d<r32float, write>;
@group(0) @binding(1) var<uniform> emitter: Emitter;
@group(0) @binding(2) var<uniform> shape: Shape;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let axis = emitter.axis;
    var dims = emitter.dims;
    dims[axis] = dims[axis] + 1u;
    if (any(gid >= dims)) {
        return;
    }
    var offset = vec3<f32>(0.5);
    offset[axis] = 0.0;
    let x = (vec3<f32>(gid) + offset) * emitter.dx;
    let v = emitter.velocity + shape_velocity(x);
    textureStore(face, vec3<i32>(gid), vec4<f32>(v[axis], 0.0, 0.0, 0.0));
}
```

- [ ] **Step 4: `shape_emitter.rs`**

Create `crates/elements-ember/src/shape_emitter.rs`:

```rust
//! `ember.emitter`: a keyframed sphere or box that emits density,
//! temperature and velocity (2b-2 spec §2.2).

use elements_core::gpu::{Axis, ComputeBatch, Field, GpuContext, GpuError, PipelineCache, StaggeredField};
use elements_core::graph::{DocError, EvalCtx, Node, NodeError, SocketSpec, SocketType, Value};
use serde::Deserialize;

use crate::kernels::{Bind, axis_index, bind_group, expect_dims, uniform_buffer};
use crate::node_util::acquire_cells;
use crate::params;
use crate::transform::{Pose, Shape, ShapeGpu, Transform};

pub const KIND: &str = "ember.emitter";

const CELLS_WGSL: &str = concat!(
    include_str!("kernels/shaders/emitter_common.wgsl"),
    include_str!("kernels/shaders/shape.wgsl"),
    include_str!("kernels/shaders/emitter_cells.wgsl"),
);

const FACES_WGSL: &str = concat!(
    include_str!("kernels/shaders/emitter_common.wgsl"),
    include_str!("kernels/shaders/shape.wgsl"),
    include_str!("kernels/shaders/emitter_faces.wgsl"),
);

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmitterParams {
    pub shape: Shape,
    pub transform: Transform,
    /// Density added per second where fully occupied.
    #[serde(default)]
    pub density_rate: f32,
    /// Temperature added per second where fully occupied.
    #[serde(default)]
    pub temperature_rate: f32,
    /// Target velocity, m/s, world space. The emitter's own motion is added.
    #[serde(default)]
    pub velocity: [f32; 3],
    /// How fast the fluid is pulled to the target velocity, 1/s; 0 turns
    /// velocity emission off (spec §3.3).
    #[serde(default)]
    pub velocity_blend: f32,
}

impl EmitterParams {
    /// No emission at all: rates, velocity and blend are zero.
    pub fn new(shape: Shape, transform: Transform) -> Self {
        Self {
            shape,
            transform,
            density_rate: 0.0,
            temperature_rate: 0.0,
            velocity: [0.0; 3],
            velocity_blend: 0.0,
        }
    }
}

/// An emitter's four outputs.
pub struct EmitterFields<'a> {
    pub density: &'a Field,
    pub temperature: &'a Field,
    /// Occupancy × velocity_blend, 1/s.
    pub weight: &'a Field,
    pub velocity: &'a StaggeredField,
}

impl EmitterFields<'_> {
    /// Every field must match the domain `density` is sized for.
    pub(crate) fn check(&self, what: &str) -> Result<(), GpuError> {
        let cells = self.density.dims();
        expect_dims(what, self.temperature, cells)?;
        expect_dims(what, self.weight, cells)?;
        if self.velocity.cells() != cells {
            return Err(GpuError::Validation(format!(
                "{what}: velocity {:?}, domain {cells:?}",
                self.velocity.cells()
            )));
        }
        Ok(())
    }
}

/// Matches `Emitter` in `emitter_common.wgsl`, 80 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EmitterGpu {
    dims: [u32; 3],
    dx: f32,
    velocity: [f32; 3],
    axis: u32,
    density_rate: f32,
    temperature_rate: f32,
    velocity_blend: f32,
    noise_on: u32,
    noise_scale: f32,
    noise_amplitude: f32,
    noise_w: f32,
    seed_lo: u32,
    seed_hi: u32,
    _pad: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<EmitterGpu>() == 80);

impl EmitterGpu {
    fn new(p: &EmitterParams, cells: [u32; 3], dx: f32, _seconds: f64) -> Self {
        Self {
            dims: cells,
            dx,
            velocity: p.velocity,
            axis: 0,
            density_rate: p.density_rate,
            temperature_rate: p.temperature_rate,
            velocity_blend: p.velocity_blend,
            noise_on: 0,
            noise_scale: 1.0,
            noise_amplitude: 0.0,
            noise_w: 0.0,
            seed_lo: 0,
            seed_hi: 0,
            _pad: [0; 3],
        }
    }
}

/// Write `params`' outputs at `pose` and `seconds` into `out`, in a domain
/// with voxel edge `dx`. Submits its own batch.
pub fn fill_emitter(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    params: &EmitterParams,
    pose: &Pose,
    seconds: f64,
    dx: f32,
    out: EmitterFields<'_>,
) -> Result<(), GpuError> {
    out.check("fill_emitter")?;
    let cells = out.density.dims();
    let cell_pipe = cache.get_or_create(gpu, "ember.emitter.cells", CELLS_WGSL, "main")?;
    let face_pipe = cache.get_or_create(gpu, "ember.emitter.faces", FACES_WGSL, "main")?;
    let shape = uniform_buffer(gpu, "ember-shape", bytemuck::bytes_of(&ShapeGpu::new(&params.shape, pose)))?;
    let base = EmitterGpu::new(params, [cells.x, cells.y, cells.z], dx, seconds);
    let cell_params = uniform_buffer(gpu, "ember-emitter", bytemuck::bytes_of(&base))?;
    let mut batch = ComputeBatch::new();
    let group = bind_group(
        gpu,
        &cell_pipe,
        &[
            Bind::Tex(out.density),
            Bind::Tex(out.temperature),
            Bind::Tex(out.weight),
            Bind::Buf(&cell_params),
            Bind::Buf(&shape),
        ],
    )?;
    batch.dispatch(&cell_pipe, &group, cells);
    for axis in Axis::ALL {
        let face_params = uniform_buffer(
            gpu,
            "ember-emitter",
            bytemuck::bytes_of(&EmitterGpu { axis: axis_index(axis), ..base }),
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
pub struct Emitter {
    params: EmitterParams,
}

impl Node for Emitter {
    fn kind(&self) -> &'static str {
        KIND
    }

    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![],
            outputs: vec![
                SocketType::Field,
                SocketType::Field,
                SocketType::Field,
                SocketType::VectorField,
            ],
        }
    }

    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let time = ctx.time();
        let pose = self.params.transform.pose(f64::from(time.frame), time.dt);
        let dx = ctx.voxel_size();
        let cells = acquire_cells(ctx, 3)?;
        let velocity = match ctx.acquire_vector_uninit() {
            Ok(v) => v,
            Err(e) => {
                for field in cells {
                    ctx.release(Value::Field(field));
                }
                return Err(e);
            }
        };
        let params = &self.params;
        let filled = ctx.with_gpu(|gpu, cache| {
            fill_emitter(
                gpu,
                cache,
                params,
                &pose,
                time.seconds,
                dx,
                EmitterFields {
                    density: &cells[0],
                    temperature: &cells[1],
                    weight: &cells[2],
                    velocity: &velocity,
                },
            )
        });
        let mut values: Vec<Value> = cells.into_iter().map(Value::Field).collect();
        values.push(Value::VectorField(velocity));
        if let Err(e) = filled {
            for value in values {
                ctx.release(value);
            }
            return Err(e);
        }
        Ok(values)
    }
}

pub(crate) fn build(params: &serde_json::Value) -> Result<Box<dyn Node>, DocError> {
    let p: EmitterParams = params::parse(KIND, params)?;
    p.shape.validate(KIND)?;
    p.transform.validate(KIND)?;
    params::finite(
        KIND,
        "rates and velocity",
        &[
            p.density_rate,
            p.temperature_rate,
            p.velocity_blend,
            p.velocity[0],
            p.velocity[1],
            p.velocity[2],
        ],
    )?;
    if p.velocity_blend < 0.0 {
        return Err(params::bad(KIND, "velocity_blend must be at least 0"));
    }
    Ok(Box::new(Emitter { params: p }))
}
```

- [ ] **Step 5: The union**

Create `crates/elements-ember/src/kernels/shaders/weights.wgsl`:

```wgsl
// The velocity weight on face `p` of `axis`'s grid: the larger weight of the
// two cells it separates, with cells beyond the domain clamped in
// (2b-2 spec §2.4, §3.3).
fn face_weight(w: texture_3d<f32>, axis: u32, p: vec3<i32>, dims: vec3<u32>) -> f32 {
    var e = vec3<i32>(0);
    e[axis] = 1;
    let last = vec3<i32>(dims) - vec3<i32>(1);
    let below = textureLoad(w, clamp(p - e, vec3<i32>(0), last), 0).x;
    let above = textureLoad(w, clamp(p, vec3<i32>(0), last), 0).x;
    return max(below, above);
}
```

Create `crates/elements-ember/src/kernels/shaders/emitter_union_cells.wgsl`:

```wgsl
// ember.emitter_union, cell pass: rates add, the weight is the maximum.

struct Grid {
    dims: vec3<u32>,
    axis: u32,
};

@group(0) @binding(0) var d1: texture_3d<f32>;
@group(0) @binding(1) var t1: texture_3d<f32>;
@group(0) @binding(2) var w1: texture_3d<f32>;
@group(0) @binding(3) var d2: texture_3d<f32>;
@group(0) @binding(4) var t2: texture_3d<f32>;
@group(0) @binding(5) var w2: texture_3d<f32>;
@group(0) @binding(6) var density: texture_storage_3d<r32float, write>;
@group(0) @binding(7) var temperature: texture_storage_3d<r32float, write>;
@group(0) @binding(8) var weight: texture_storage_3d<r32float, write>;
@group(0) @binding(9) var<uniform> grid: Grid;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= grid.dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    let d = textureLoad(d1, p, 0).x + textureLoad(d2, p, 0).x;
    let t = textureLoad(t1, p, 0).x + textureLoad(t2, p, 0).x;
    let w = max(textureLoad(w1, p, 0).x, textureLoad(w2, p, 0).x);
    textureStore(density, p, vec4<f32>(d, 0.0, 0.0, 0.0));
    textureStore(temperature, p, vec4<f32>(t, 0.0, 0.0, 0.0));
    textureStore(weight, p, vec4<f32>(w, 0.0, 0.0, 0.0));
}
```

Create `crates/elements-ember/src/kernels/shaders/emitter_union_faces.wgsl`:

```wgsl
// ember.emitter_union, face pass: the target velocity, weighted by each
// emitter's face weight.

struct Grid {
    dims: vec3<u32>,
    axis: u32,
};

@group(0) @binding(0) var w1: texture_3d<f32>;
@group(0) @binding(1) var w2: texture_3d<f32>;
@group(0) @binding(2) var u1: texture_3d<f32>;
@group(0) @binding(3) var u2: texture_3d<f32>;
@group(0) @binding(4) var dst: texture_storage_3d<r32float, write>;
@group(0) @binding(5) var<uniform> grid: Grid;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let axis = grid.axis;
    var dims = grid.dims;
    dims[axis] = dims[axis] + 1u;
    if (any(gid >= dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    let a = face_weight(w1, axis, p, grid.dims);
    let b = face_weight(w2, axis, p, grid.dims);
    let u = (a * textureLoad(u1, p, 0).x + b * textureLoad(u2, p, 0).x) / max(a + b, 1e-12);
    textureStore(dst, p, vec4<f32>(u, 0.0, 0.0, 0.0));
}
```

Create `crates/elements-ember/src/unions.rs`:

```rust
//! `ember.emitter_union` (and, from Task 4, `ember.collider_union`): merge two
//! emitters or two colliders into one (2b-2 spec §2.4). Chain them for more.

use elements_core::gpu::{Axis, ComputeBatch, GpuContext, GpuError, PipelineCache};
use elements_core::graph::{DocError, EvalCtx, Node, NodeError, SocketSpec, SocketType, Value};

use crate::kernels::{Bind, axis_index, bind_group, uniform_buffer};
use crate::node_util::{acquire_cells, take_inputs};
use crate::shape_emitter::EmitterFields;

pub const EMITTER_UNION_KIND: &str = "ember.emitter_union";

const EMITTER_CELLS_WGSL: &str = include_str!("kernels/shaders/emitter_union_cells.wgsl");
const EMITTER_FACES_WGSL: &str = concat!(
    include_str!("kernels/shaders/weights.wgsl"),
    include_str!("kernels/shaders/emitter_union_faces.wgsl"),
);

/// Matches `Grid` in the union shaders, 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GridGpu {
    dims: [u32; 3],
    axis: u32,
}

const _: () = assert!(std::mem::size_of::<GridGpu>() == 16);

/// `out` = the union of `a` and `b`. Submits its own batch.
pub fn union_emitters(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    a: EmitterFields<'_>,
    b: EmitterFields<'_>,
    out: EmitterFields<'_>,
) -> Result<(), GpuError> {
    a.check("union_emitters a")?;
    b.check("union_emitters b")?;
    out.check("union_emitters out")?;
    let cells = out.density.dims();
    if a.density.dims() != cells || b.density.dims() != cells {
        return Err(GpuError::Validation("union_emitters: inputs differ in size".to_owned()));
    }
    let dims = [cells.x, cells.y, cells.z];
    let cell_pipe = cache.get_or_create(gpu, "ember.emitter_union.cells", EMITTER_CELLS_WGSL, "main")?;
    let face_pipe = cache.get_or_create(gpu, "ember.emitter_union.faces", EMITTER_FACES_WGSL, "main")?;
    let grid = uniform_buffer(gpu, "ember-union", bytemuck::bytes_of(&GridGpu { dims, axis: 0 }))?;
    let mut batch = ComputeBatch::new();
    let group = bind_group(
        gpu,
        &cell_pipe,
        &[
            Bind::Tex(a.density),
            Bind::Tex(a.temperature),
            Bind::Tex(a.weight),
            Bind::Tex(b.density),
            Bind::Tex(b.temperature),
            Bind::Tex(b.weight),
            Bind::Tex(out.density),
            Bind::Tex(out.temperature),
            Bind::Tex(out.weight),
            Bind::Buf(&grid),
        ],
    )?;
    batch.dispatch(&cell_pipe, &group, cells);
    for axis in Axis::ALL {
        let grid = uniform_buffer(
            gpu,
            "ember-union",
            bytemuck::bytes_of(&GridGpu { dims, axis: axis_index(axis) }),
        )?;
        let dst = out.velocity.face(axis);
        let group = bind_group(
            gpu,
            &face_pipe,
            &[
                Bind::Tex(a.weight),
                Bind::Tex(b.weight),
                Bind::Tex(a.velocity.face(axis)),
                Bind::Tex(b.velocity.face(axis)),
                Bind::Tex(dst),
                Bind::Buf(&grid),
            ],
        )?;
        batch.dispatch(&face_pipe, &group, dst.dims());
    }
    batch.submit(gpu)
}

/// The four emitter outputs of `values[at..at + 4]`, type-checked.
fn emitter_fields(values: &[Value], at: usize) -> Result<EmitterFields<'_>, NodeError> {
    Ok(EmitterFields {
        density: values[at].as_field()?,
        temperature: values[at + 1].as_field()?,
        weight: values[at + 2].as_field()?,
        velocity: values[at + 3].as_vector_field()?,
    })
}

#[derive(Debug, Clone)]
pub struct EmitterUnion;

impl Node for EmitterUnion {
    fn kind(&self) -> &'static str {
        EMITTER_UNION_KIND
    }

    fn sockets(&self) -> SocketSpec {
        let group = [
            SocketType::Field,
            SocketType::Field,
            SocketType::Field,
            SocketType::VectorField,
        ];
        SocketSpec {
            inputs: [group, group].concat(),
            outputs: group.to_vec(),
        }
    }

    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let inputs = take_inputs(ctx, 8)?;
        let result = union_node(ctx, &inputs);
        for value in inputs {
            ctx.release(value);
        }
        result
    }
}

fn union_node(ctx: &mut EvalCtx<'_>, inputs: &[Value]) -> Result<Vec<Value>, NodeError> {
    let (a, b) = (emitter_fields(inputs, 0)?, emitter_fields(inputs, 4)?);
    let cells = acquire_cells(ctx, 3)?;
    let velocity = match ctx.acquire_vector_uninit() {
        Ok(v) => v,
        Err(e) => {
            for field in cells {
                ctx.release(Value::Field(field));
            }
            return Err(e);
        }
    };
    let merged = ctx.with_gpu(|gpu, cache| {
        union_emitters(
            gpu,
            cache,
            a,
            b,
            EmitterFields {
                density: &cells[0],
                temperature: &cells[1],
                weight: &cells[2],
                velocity: &velocity,
            },
        )
    });
    let mut values: Vec<Value> = cells.into_iter().map(Value::Field).collect();
    values.push(Value::VectorField(velocity));
    if let Err(e) = merged {
        for value in values {
            ctx.release(value);
        }
        return Err(e);
    }
    Ok(values)
}

pub(crate) fn build_emitter_union(params: &serde_json::Value) -> Result<Box<dyn Node>, DocError> {
    if !(params.is_null() || params.as_object().is_some_and(|o| o.is_empty())) {
        return Err(crate::params::bad(EMITTER_UNION_KIND, "takes no parameters"));
    }
    Ok(Box::new(EmitterUnion))
}
```

If the borrow checker rejects holding `a` and `b` (borrowed from `inputs`) across `ctx.acquire_*` calls, it will be because of `ctx`, not `inputs`. `inputs` is a separate slice, so this should compile. If it doesn't, acquire the outputs before building `a` and `b`.

In `lib.rs`, add `pub mod node_util;` (or `mod node_util;`), `pub mod shape_emitter;` and `pub mod unions;`, and register them in `register`:

```rust
    registry.register(shape_emitter::KIND, shape_emitter::build);
    registry.register(unions::EMITTER_UNION_KIND, unions::build_emitter_union);
```

Remove the Task 1 `#[allow(dead_code)]`s from `transform.rs`.

- [ ] **Step 6: Correct the spec**

In `docs/superpowers/specs/2026-09-23-ember-scene-content-2b2-design.md`:
- §2.2 output 3 becomes "velocity weight (Field, occupancy × `velocity_blend`, in 1/s)". Replace the sentence about occupancy with: "Occupancy `clamp(0.5 − sdf/dx, 0, 1)` uses the one-voxel smoothed edge `ember.sphere_emitter` already uses. The rates are occupancy × rate × the noise factor, and the weight is occupancy × `velocity_blend`."
- §2.4 emitter union: "…the weight is the maximum, and the emitted velocity is `(a·u₁ + b·u₂) / max(a + b, ε)`, where `a` and `b` are each emitter's face weight: the larger weight of the face's two cells."
- §3.1 socket 2 becomes "velocity weight".
- §3.3 formula becomes `u ← u + (u_e − u) · (1 − exp(−w_face · h))`, where `w_face` is the face weight.
- Add one sentence explaining why: a union of emitters with different blend rates cannot be expressed through occupancy, so the weight carries the rate.

- [ ] **Step 7: Run and prove**

Run: `cargo nextest run -p elements-ember --test shape_emitter`
Expected: 5 passed.

| Test | Mutation | Expected |
|---|---|---|
| `rotation_turns_the_box` | in `shape_sdf`, `let p = x - shape.origin;` (drop the rotation) | FAIL: "along +y" |
| `a_box_emits_its_volume_at_any_resolution_and_orientation` | in `emitter_cells.wgsl`, `0.5 - shape_sdf(x) / emitter.dx` becomes `0.5 - shape_sdf(x) / (2.0 * emitter.dx)` | FAIL at 32³ or 64³ |
| `target_velocity_carries_the_emitters_motion_and_weight_is_occupancy_times_blend` | in `shape_velocity`, return `shape.linear` | FAIL on a face |
| `a_union_adds_rates_takes_the_max_weight_and_weights_velocity` | in `emitter_union_cells.wgsl`, `max` for the density sum | FAIL "density" |
| `bad_emitter_parameters_are_rejected` | delete the `velocity_blend < 0.0` check | FAIL |

- [ ] **Step 8: Run the gate and commit**

Run: `just check`
Expected: PASS.

```bash
git add crates/elements-ember docs/superpowers/specs/2026-09-23-ember-scene-content-2b2-design.md
git commit   # subject: "Add the keyframed box and sphere emitter, and emitter unions"
```

The body explains why:
- Emitters are fields (umbrella E4), so they compose through union nodes.
- The third output is a weight rather than occupancy, and the spec is corrected to match.

Include the mutation outputs.

---
## Task 3: Noise-modulated emission

Spec §2.2 (Noise).

**Files:**
- Modify: `crates/elements-ember/src/shape_emitter.rs` (`Noise`, `EmitterParams::noise`, `EmitterGpu::new`, validation)
- Modify: `crates/elements-ember/src/kernels/shaders/emitter_cells.wgsl` (`noise_factor`)
- Test: `crates/elements-ember/tests/noise_emitter.rs` (create)

**Interfaces:**
- Consumes: `EmitterParams`, `fill_emitter` and the `run` pattern (Task 2).
- Produces: `shape_emitter::Noise { seed: u64, scale_m: f32, amplitude: f32, evolution: f32 }`, and `EmitterParams::noise: Option<Noise>` (default `None`, set to `None` in `EmitterParams::new`).

- [ ] **Step 1: Write the failing tests**

Create `crates/elements-ember/tests/noise_emitter.rs`:

```rust
mod common;

use common::*;
use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, GpuContext, PipelineCache};
use elements_ember::shape_emitter::{EmitterFields, EmitterParams, Noise, fill_emitter};
use elements_ember::transform::{Shape, Transform};

const SPF: f64 = 1.0 / 24.0;

/// A box filling the whole 2 m domain, density rate 1, with `noise`; the
/// density output is then exactly the noise factor in every cell.
fn factors(gpu: &GpuContext, n: u32, noise: Noise, seconds: f64) -> Vec<f32> {
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(n, n, n);
    let dx = 2.0 / n as f32;
    let p = EmitterParams {
        density_rate: 1.0,
        noise: Some(noise),
        ..EmitterParams::new(Shape::Box { half_extents: [2.0; 3] }, Transform::at([1.0; 3]))
    };
    let f: [_; 3] = std::array::from_fn(|_| pool.acquire(gpu, cells, FieldFormat::R32Float).unwrap());
    let v = pool.acquire_staggered_uninit(gpu, cells).unwrap();
    let pose = p.transform.pose(seconds / SPF, SPF);
    fill_emitter(gpu, &mut cache, &p, &pose, seconds, dx, EmitterFields {
        density: &f[0],
        temperature: &f[1],
        weight: &f[2],
        velocity: &v,
    })
    .unwrap();
    f[0].read_back(gpu).unwrap()
}

fn noise(seed: u64, evolution: f32) -> Noise {
    Noise { seed, scale_m: 0.1, amplitude: 0.8, evolution }
}

/// Spec §2.2: the same seed gives the same pattern, bit for bit, and
/// another seed does not.
#[test]
fn noise_is_deterministic_for_a_seed() {
    let gpu = gpu();
    let a = factors(&gpu, 32, noise(7, 0.0), 0.0);
    let b = factors(&gpu, 32, noise(7, 0.0), 0.0);
    let c = factors(&gpu, 32, noise(8, 0.0), 0.0);
    assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "same seed");
    assert!(a != c, "another seed");
}

/// Value noise averages 0.5, so the factor 1 − amplitude·(1 − n) averages
/// 1 − amplitude/2, and it stays within [1 − amplitude, 1].
#[test]
fn the_mean_factor_is_one_minus_half_the_amplitude() {
    let gpu = gpu();
    let f = factors(&gpu, 32, noise(3, 0.0), 0.0);
    let mean = f.iter().map(|&v| f64::from(v)).sum::<f64>() / f.len() as f64;
    assert!((mean - 0.6).abs() <= 0.03, "mean {mean}");
    assert!(f.iter().all(|&v| (0.2 - 1e-6..=1.0 + 1e-6).contains(&v)), "range");
}

/// With `evolution` > 0 the pattern changes over time; with 0 it does not.
#[test]
fn noise_evolves_only_when_asked() {
    let gpu = gpu();
    let still = (factors(&gpu, 32, noise(5, 0.0), 0.0), factors(&gpu, 32, noise(5, 0.0), 1.0));
    assert!(still.0 == still.1, "evolution 0 must not change");
    let moving = (factors(&gpu, 32, noise(5, 2.0), 0.0), factors(&gpu, 32, noise(5, 2.0), 1.0));
    let changed = moving.0.iter().zip(&moving.1).filter(|(a, b)| (**a - **b).abs() > 1e-3).count();
    assert!(changed * 2 > moving.0.len(), "only {changed} cells changed");
}

/// Noise is in metres, so the pattern is the same at any resolution: each
/// 32³ cell matches the mean of the eight 64³ cells it covers, within the
/// smoothing that averaging adds.
#[test]
fn the_pattern_does_not_depend_on_resolution() {
    let gpu = gpu();
    let n = Noise { seed: 11, scale_m: 0.5, amplitude: 1.0, evolution: 0.0 };
    let coarse = factors(&gpu, 32, n, 0.0);
    let fine = factors(&gpu, 64, n, 0.0);
    let (c, f) = (FieldDims::new(32, 32, 32), FieldDims::new(64, 64, 64));
    let mut worst = 0.0f32;
    for k in 0..32 {
        for j in 0..32 {
            for i in 0..32 {
                let mut sum = 0.0;
                for dz in 0..2 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            sum += fine[index(f, 2 * i + dx, 2 * j + dy, 2 * k + dz)];
                        }
                    }
                }
                worst = worst.max((sum / 8.0 - coarse[index(c, i, j, k)]).abs());
            }
        }
    }
    assert!(worst <= 0.05, "worst difference {worst}");
}

#[test]
fn bad_noise_is_rejected() {
    let base = serde_json::json!({ "shape": { "sphere": { "radius": 0.2 } }, "transform": { "keys": [{ "frame": 0 }] } });
    let with = |noise: serde_json::Value| {
        let mut o = base.clone();
        o["noise"] = noise;
        elements_ember::registry().build(elements_ember::shape_emitter::KIND, &o).is_err()
    };
    assert!(!with(serde_json::json!({ "seed": 1, "scale_m": 0.1, "amplitude": 0.5 })));
    assert!(with(serde_json::json!({ "seed": 1, "scale_m": 0.0, "amplitude": 0.5 })));
    assert!(with(serde_json::json!({ "seed": 1, "scale_m": 0.1, "amplitude": 1.5 })));
    assert!(with(serde_json::json!({ "scale_m": 0.1, "amplitude": 0.5 })), "seed is required");
}
```

Run: `cargo nextest run -p elements-ember --test noise_emitter`
Expected: compile error, `Noise` not found.

- [ ] **Step 2: Parameters**

In `shape_emitter.rs`, add:

```rust
/// Noise that modulates an emitter's rates (spec §2.2).
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Noise {
    pub seed: u64,
    /// Size of one noise cell, metres.
    pub scale_m: f32,
    /// 0 turns the noise off; 1 lets it cut emission to zero.
    pub amplitude: f32,
    /// How fast the pattern changes, per second; 0 is a still pattern.
    #[serde(default)]
    pub evolution: f32,
}
```

Add to `EmitterParams`, after `velocity_blend`:

```rust
    /// Optional noise that multiplies the rates.
    #[serde(default)]
    pub noise: Option<Noise>,
```

and `noise: None` in `EmitterParams::new`. Replace `EmitterGpu::new` with:

```rust
    fn new(p: &EmitterParams, cells: [u32; 3], dx: f32, seconds: f64) -> Self {
        let noise = p.noise.unwrap_or(Noise { seed: 0, scale_m: 1.0, amplitude: 0.0, evolution: 0.0 });
        Self {
            dims: cells,
            dx,
            velocity: p.velocity,
            axis: 0,
            density_rate: p.density_rate,
            temperature_rate: p.temperature_rate,
            velocity_blend: p.velocity_blend,
            noise_on: u32::from(p.noise.is_some()),
            noise_scale: noise.scale_m,
            noise_amplitude: noise.amplitude,
            // Time enters only through this coordinate, so a frame's pattern
            // depends on the document's time alone.
            noise_w: (f64::from(noise.evolution) * seconds) as f32,
            seed_lo: noise.seed as u32,
            seed_hi: (noise.seed >> 32) as u32,
            _pad: [0; 3],
        }
    }
```

In `build`, after the existing checks:

```rust
    if let Some(n) = p.noise {
        params::finite(KIND, "noise", &[n.scale_m, n.amplitude, n.evolution])?;
        if n.scale_m <= 0.0 {
            return Err(params::bad(KIND, "noise scale_m must be positive"));
        }
        if !(0.0..=1.0).contains(&n.amplitude) {
            return Err(params::bad(KIND, "noise amplitude must be in [0, 1]"));
        }
    }
```

- [ ] **Step 3: The noise function**

In `emitter_cells.wgsl`, replace `noise_factor` with:

```wgsl
// Integer hash of a 4D lattice point and the 64-bit seed, to [0, 1]. The
// same construction as core's noise: integer operations only, so it is
// deterministic across backends.
fn hash4(p: vec4<i32>) -> f32 {
    var h: u32 = emitter.seed_lo ^ (emitter.seed_hi * 0x27d4eb2du);
    h = h ^ (u32(p.x) * 0x9e3779b9u);
    h = h ^ (u32(p.y) * 0x85ebca6bu);
    h = h ^ (u32(p.z) * 0xc2b2ae35u);
    h = h ^ (u32(p.w) * 0x165667b1u);
    h = h ^ (h >> 15u);
    h = h * 0x2c1b3c6du;
    h = h ^ (h >> 12u);
    h = h * 0x297a2d39u;
    h = h ^ (h >> 15u);
    return f32(h & 0xffffffu) * (1.0 / 16777215.0);
}

// Value noise in 4D: the 16 lattice corners around `q`, blended with a
// smoothstep fade. Averages 0.5. Once per cell per frame, so the loop is fine.
fn value_noise(q: vec4<f32>) -> f32 {
    let base = floor(q);
    let i = vec4<i32>(base);
    let f = q - base;
    let t = f * f * (vec4<f32>(3.0) - 2.0 * f);
    var acc = 0.0;
    for (var n = 0u; n < 16u; n = n + 1u) {
        let o = vec4<u32>(n & 1u, (n >> 1u) & 1u, (n >> 2u) & 1u, (n >> 3u) & 1u);
        let w = select(vec4<f32>(1.0) - t, t, o == vec4<u32>(1u));
        acc += w.x * w.y * w.z * w.w * hash4(i + vec4<i32>(o));
    }
    return acc;
}

// The emission factor at world point `x`: 1 − amplitude · (1 − n), where `n` is
// noise in metre coordinates plus time as a fourth coordinate (spec §2.2).
fn noise_factor(x: vec3<f32>) -> f32 {
    if (emitter.noise_on == 0u) {
        return 1.0;
    }
    let n = value_noise(vec4<f32>(x / emitter.noise_scale, emitter.noise_w));
    return 1.0 - emitter.noise_amplitude * (1.0 - n);
}
```

- [ ] **Step 4: Run and prove**

Run: `cargo nextest run -p elements-ember --test noise_emitter && cargo nextest run -p elements-ember --test shape_emitter`
Expected: all pass. Task 2's tests still pass: noise defaults to off.

If `the_mean_factor_is_one_minus_half_the_amplitude` misses its ±0.03 band, or `the_pattern_does_not_depend_on_resolution` misses 0.05, stop and report the measured value. Do not change a bound.

| Test | Mutation | Expected |
|---|---|---|
| `the_pattern_does_not_depend_on_resolution` | in `noise_factor`, `x / emitter.noise_scale` becomes `x / emitter.dx / emitter.noise_scale` (grid units) | FAIL: worst difference |
| `noise_is_deterministic_for_a_seed` | `hash4` ignores the seed (drop the `emitter.seed_lo ^ …` initialiser; start `h` at 0) | FAIL: "another seed" |
| `noise_evolves_only_when_asked` | `noise_w: 0.0` in `EmitterGpu::new` | FAIL: "only 0 cells changed" |
| `the_mean_factor_is_one_minus_half_the_amplitude` | `(1.0 - n)` becomes `(1.0 - n * n)` in `noise_factor` | FAIL: mean ≈ 0.44 |
| `bad_noise_is_rejected` | delete the amplitude range check | FAIL |

- [ ] **Step 5: Run the gate and commit**

Run: `just check`
Expected: PASS.

```bash
git add crates/elements-ember
git commit   # subject: "Modulate emitters with noise in metres that evolves over time"
```

The body explains why:
- The noise is in metres, so a preview and a bake look the same.
- Time enters only through the fourth coordinate, so frames stay deterministic.

Include the mutation outputs.

---
## Task 4: `ember.collider` and `ember.collider_union`

Spec §2.3 and §2.4.

**Files:**
- Create: `crates/elements-ember/src/collider.rs`, `crates/elements-ember/src/kernels/shaders/collider_common.wgsl`, `collider_cells.wgsl`, `collider_faces.wgsl`, `collider_union_cells.wgsl`, `collider_union_faces.wgsl`
- Modify: `crates/elements-ember/src/kernels/shaders/weights.wgsl` (`face_min`), `crates/elements-ember/src/unions.rs`, `crates/elements-ember/src/lib.rs`
- Test: `crates/elements-ember/tests/collider.rs` (create)

**Interfaces:**
- Consumes: `Shape`, `Transform`, `Pose`, `ShapeGpu`, `shape.wgsl` (Task 1); `uniform_buffer`, `take_inputs`, `GridGpu` and `node_util::produce(ctx, cells, fill) -> Result<Vec<Value>, NodeError>`, which acquires `cells` R32Float fields and one staggered field, runs `fill`, and releases everything on any failure (Task 2).
- Produces:
  - `collider::KIND = "ember.collider"`
  - `ColliderParams { shape: Shape, transform: Transform }`, deriving `Serialize` and `Deserialize`. Task 7 serialises it into benchmark documents.
  - `ColliderFields<'a> { sdf: &'a Field, velocity: &'a StaggeredField }`
  - `fill_collider(gpu, cache, params: &ColliderParams, pose: &Pose, dx: f32, out: ColliderFields<'_>) -> Result<(), GpuError>`
  - `unions::COLLIDER_UNION_KIND = "ember.collider_union"` and `unions::union_colliders(gpu, cache, a: ColliderFields<'_>, b: ColliderFields<'_>, out: ColliderFields<'_>) -> Result<(), GpuError>`
  - `weights.wgsl`: `fn face_min(s: texture_3d<f32>, axis: u32, p: vec3<i32>, dims: vec3<u32>) -> f32`
- Node sockets: `ember.collider` outputs `[Field sdf, VectorField velocity]`; `ember.collider_union` takes `[F, V, F, V]` and outputs `[F, V]`.

- [ ] **Step 1: Write the failing tests**

Create `crates/elements-ember/tests/collider.rs`:

```rust
mod common;

use common::*;
use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, GpuContext, PipelineCache};
use elements_ember::collider::{ColliderFields, ColliderParams, fill_collider};
use elements_ember::transform::{Key, Pose, Rotate, Shape, Transform};
use elements_ember::unions::union_colliders;

const SPF: f64 = 1.0 / 24.0;
const CELLS: FieldDims = FieldDims { x: 12, y: 10, z: 8 };
const DX: f32 = 2.0 / 12.0;

fn run(gpu: &GpuContext, params: &ColliderParams, frame: f64) -> (Vec<f32>, [Vec<f32>; 3], Pose) {
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let sdf = pool.acquire(gpu, CELLS, FieldFormat::R32Float).unwrap();
    let v = pool.acquire_staggered_uninit(gpu, CELLS).unwrap();
    let pose = params.transform.pose(frame, SPF);
    fill_collider(gpu, &mut cache, params, &pose, DX, ColliderFields { sdf: &sdf, velocity: &v }).unwrap();
    (sdf.read_back(gpu).unwrap(), read_staggered(gpu, &v), pose)
}

/// CPU box distance, mirroring `shape_sdf` in shape.wgsl.
fn box_sdf(pose: &Pose, half: [f64; 3], x: [f64; 3]) -> f64 {
    let m = pose.world_to_local();
    let d: [f64; 3] = std::array::from_fn(|a| x[a] - pose.translate[a]);
    let p: [f64; 3] = std::array::from_fn(|r| m[r][0] * d[0] + m[r][1] * d[1] + m[r][2] * d[2]);
    let q: [f64; 3] = std::array::from_fn(|a| p[a].abs() - half[a]);
    let outside = q.iter().map(|v| v.max(0.0).powi(2)).sum::<f64>().sqrt();
    outside + q[0].max(q[1]).max(q[2]).min(0.0)
}

fn cell_centre(i: u32, j: u32, k: u32) -> [f64; 3] {
    [i, j, k].map(|c| (f64::from(c) + 0.5) * f64::from(DX))
}

/// Spec §2.3: the SDF output is the exact signed distance, negative inside,
/// for spheres and for rotated boxes.
#[test]
fn the_sdf_is_the_signed_distance_to_the_shape() {
    let gpu = gpu();
    let sphere = ColliderParams {
        shape: Shape::Sphere { radius: 0.3 },
        transform: Transform::at([1.0, 0.9, 0.7]),
    };
    let (s, _, _) = run(&gpu, &sphere, 0.0);
    let half = [0.3, 0.2, 0.25];
    let boxed = ColliderParams {
        shape: Shape::Box { half_extents: half.map(|v| v as f32) },
        transform: Transform {
            keys: vec![Key {
                frame: 0.0,
                translate: [1.0, 0.8, 0.6],
                rotate: Some(Rotate { axis: [0.0, 1.0, 1.0], degrees: 30.0 }),
            }],
        },
    };
    let (b, _, pose) = run(&gpu, &boxed, 0.0);
    let mut inside = 0;
    for k in 0..CELLS.z {
        for j in 0..CELLS.y {
            for i in 0..CELLS.x {
                let x = cell_centre(i, j, k);
                let at = index(CELLS, i, j, k);
                let r = ((x[0] - 1.0).powi(2) + (x[1] - 0.9).powi(2) + (x[2] - 0.7).powi(2)).sqrt();
                assert!((f64::from(s[at]) - (r - 0.3)).abs() <= 1e-5, "sphere {:?}", [i, j, k]);
                let want = box_sdf(&pose, half, x);
                assert!((f64::from(b[at]) - want).abs() <= 1e-5, "box {:?}: {} vs {want}", [i, j, k], b[at]);
                inside += usize::from(want < 0.0);
            }
        }
    }
    assert!(inside > 0, "some cell must be inside the box");
}

/// Spec §2.3: the velocity output is v + ω × r at every face centre.
#[test]
fn the_velocity_is_the_colliders_material_velocity_at_each_face() {
    let gpu = gpu();
    let spinning = ColliderParams {
        shape: Shape::Box { half_extents: [0.2, 0.2, 0.2] },
        transform: Transform {
            keys: vec![
                Key { frame: 0.0, translate: [0.8, 0.8, 0.6], rotate: None },
                Key {
                    frame: 24.0,
                    translate: [1.2, 0.9, 0.6],
                    rotate: Some(Rotate { axis: [0.0, 0.0, 1.0], degrees: 90.0 }),
                },
            ],
        },
    };
    let (_, faces, pose) = run(&gpu, &spinning, 6.0);
    for a in 0..3 {
        let d = face_dims(CELLS, a);
        let off = face_offset(a);
        for k in 0..d.z {
            for j in 0..d.y {
                for i in 0..d.x {
                    let x = [i as f32 + off[0], j as f32 + off[1], k as f32 + off[2]]
                        .map(|c| f64::from(c) * f64::from(DX));
                    let want = pose.velocity_at(x)[a];
                    let got = f64::from(faces[a][index(d, i, j, k)]);
                    assert!((got - want).abs() <= 1e-5, "face {a} {:?}: {got} vs {want}", [i, j, k]);
                }
            }
        }
    }
}

/// Mirrors `face_min` in weights.wgsl.
fn face_min(s: &[f32], axis: usize, p: [u32; 3]) -> f32 {
    let n = [CELLS.x as i32, CELLS.y as i32, CELLS.z as i32];
    let at = |q: [i32; 3]| {
        let c: [u32; 3] = std::array::from_fn(|a| q[a].clamp(0, n[a] - 1) as u32);
        s[index(CELLS, c[0], c[1], c[2])]
    };
    let mut below = p.map(|v| v as i32);
    below[axis] -= 1;
    at(below).min(at(p.map(|v| v as i32)))
}

/// Spec §2.4: the union's SDF is the minimum, and each face takes the
/// velocity of the collider nearer to it.
#[test]
fn a_collider_union_takes_the_nearer_colliders_distance_and_velocity() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let moving = |from: [f32; 3], to: [f32; 3]| Transform {
        keys: vec![
            Key { frame: 0.0, translate: from, rotate: None },
            Key { frame: 24.0, translate: to, rotate: None },
        ],
    };
    let a = ColliderParams { shape: Shape::Sphere { radius: 0.3 }, transform: moving([0.6, 0.8, 0.6], [0.6, 0.8, 1.0]) };
    let b = ColliderParams { shape: Shape::Box { half_extents: [0.25; 3] }, transform: moving([1.4, 0.8, 0.6], [1.0, 0.8, 0.6]) };
    let make = |pool: &mut FieldPool| {
        (
            pool.acquire(&gpu, CELLS, FieldFormat::R32Float).unwrap(),
            pool.acquire_staggered_uninit(&gpu, CELLS).unwrap(),
        )
    };
    let (sa, va) = make(&mut pool);
    let (sb, vb) = make(&mut pool);
    let (so, vo) = make(&mut pool);
    for (p, s, v) in [(&a, &sa, &va), (&b, &sb, &vb)] {
        let pose = p.transform.pose(6.0, SPF);
        fill_collider(&gpu, &mut cache, p, &pose, DX, ColliderFields { sdf: s, velocity: v }).unwrap();
    }
    union_colliders(
        &gpu,
        &mut cache,
        ColliderFields { sdf: &sa, velocity: &va },
        ColliderFields { sdf: &sb, velocity: &vb },
        ColliderFields { sdf: &so, velocity: &vo },
    )
    .unwrap();
    let (ca, cb, co) = (sa.read_back(&gpu).unwrap(), sb.read_back(&gpu).unwrap(), so.read_back(&gpu).unwrap());
    for n in 0..CELLS.voxel_count() {
        assert_eq!(co[n], ca[n].min(cb[n]), "sdf {n}");
    }
    let (ua, ub, uo) = (read_staggered(&gpu, &va), read_staggered(&gpu, &vb), read_staggered(&gpu, &vo));
    for ax in 0..3 {
        let d = face_dims(CELLS, ax);
        for k in 0..d.z {
            for j in 0..d.y {
                for i in 0..d.x {
                    let at = index(d, i, j, k);
                    let nearer_a = face_min(&ca, ax, [i, j, k]) <= face_min(&cb, ax, [i, j, k]);
                    let want = if nearer_a { ua[ax][at] } else { ub[ax][at] };
                    assert_eq!(uo[ax][at], want, "face {ax} {:?}", [i, j, k]);
                }
            }
        }
    }
}

#[test]
fn bad_collider_parameters_are_rejected() {
    let build = |p: serde_json::Value| elements_ember::registry().build(elements_ember::collider::KIND, &p).is_err();
    assert!(!build(serde_json::json!({ "shape": { "sphere": { "radius": 0.2 } }, "transform": { "keys": [{ "frame": 0 }] } })));
    assert!(build(serde_json::json!({ "shape": { "sphere": { "radius": -0.2 } }, "transform": { "keys": [{ "frame": 0 }] } })));
    assert!(build(serde_json::json!({ "shape": { "sphere": { "radius": 0.2 } } })), "transform is required");
    assert!(build(serde_json::json!({ "shape": { "sphere": { "radius": 0.2 } }, "transform": { "keys": [{ "frame": 0 }] }, "density_rate": 1.0 })));
}
```

Run: `cargo nextest run -p elements-ember --test collider`
Expected: compile error, no module `collider`.

- [ ] **Step 2: The collider shaders**

Create `crates/elements-ember/src/kernels/shaders/collider_common.wgsl`:

```wgsl
// Shared by the collider's cell and face passes (2b-2 spec §2.3). 32 bytes;
// mirrors `ColliderGpu` in collider.rs.

struct Collider {
    dims: vec3<u32>,
    dx: f32,
    axis: u32,   // the face pass's axis (0, 1, 2)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
```

Create `crates/elements-ember/src/kernels/shaders/collider_cells.wgsl`:

```wgsl
// ember.collider, cell pass: the signed distance at every cell centre, metres.

@group(0) @binding(0) var sdf: texture_storage_3d<r32float, write>;
@group(0) @binding(1) var<uniform> collider: Collider;
@group(0) @binding(2) var<uniform> shape: Shape;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= collider.dims)) {
        return;
    }
    let x = (vec3<f32>(gid) + vec3<f32>(0.5)) * collider.dx;
    textureStore(sdf, vec3<i32>(gid), vec4<f32>(shape_sdf(x), 0.0, 0.0, 0.0));
}
```

Create `crates/elements-ember/src/kernels/shaders/collider_faces.wgsl`:

```wgsl
// ember.collider, face pass: the collider's material velocity at every face
// centre of one axis, m/s. Defined everywhere; the solver uses it only on
// faces that touch a solid cell.

@group(0) @binding(0) var face: texture_storage_3d<r32float, write>;
@group(0) @binding(1) var<uniform> collider: Collider;
@group(0) @binding(2) var<uniform> shape: Shape;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let axis = collider.axis;
    var dims = collider.dims;
    dims[axis] = dims[axis] + 1u;
    if (any(gid >= dims)) {
        return;
    }
    var offset = vec3<f32>(0.5);
    offset[axis] = 0.0;
    let x = (vec3<f32>(gid) + offset) * collider.dx;
    textureStore(face, vec3<i32>(gid), vec4<f32>(shape_velocity(x)[axis], 0.0, 0.0, 0.0));
}
```

Append to `weights.wgsl`:

```wgsl

// The smaller signed distance of the two cells face `p` of `axis`'s grid
// separates, with cells beyond the domain clamped in (2b-2 spec §2.4).
fn face_min(s: texture_3d<f32>, axis: u32, p: vec3<i32>, dims: vec3<u32>) -> f32 {
    var e = vec3<i32>(0);
    e[axis] = 1;
    let last = vec3<i32>(dims) - vec3<i32>(1);
    let below = textureLoad(s, clamp(p - e, vec3<i32>(0), last), 0).x;
    let above = textureLoad(s, clamp(p, vec3<i32>(0), last), 0).x;
    return min(below, above);
}
```

Create `crates/elements-ember/src/kernels/shaders/collider_union_cells.wgsl`:

```wgsl
// ember.collider_union, cell pass: the smaller signed distance.

struct Grid {
    dims: vec3<u32>,
    axis: u32,
};

@group(0) @binding(0) var s1: texture_3d<f32>;
@group(0) @binding(1) var s2: texture_3d<f32>;
@group(0) @binding(2) var sdf: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var<uniform> grid: Grid;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= grid.dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    textureStore(sdf, p, vec4<f32>(min(textureLoad(s1, p, 0).x, textureLoad(s2, p, 0).x), 0.0, 0.0, 0.0));
}
```

Create `crates/elements-ember/src/kernels/shaders/collider_union_faces.wgsl`:

```wgsl
// ember.collider_union, face pass: the velocity of whichever collider is
// nearer to the face.

struct Grid {
    dims: vec3<u32>,
    axis: u32,
};

@group(0) @binding(0) var s1: texture_3d<f32>;
@group(0) @binding(1) var s2: texture_3d<f32>;
@group(0) @binding(2) var u1: texture_3d<f32>;
@group(0) @binding(3) var u2: texture_3d<f32>;
@group(0) @binding(4) var dst: texture_storage_3d<r32float, write>;
@group(0) @binding(5) var<uniform> grid: Grid;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let axis = grid.axis;
    var dims = grid.dims;
    dims[axis] = dims[axis] + 1u;
    if (any(gid >= dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    var u = textureLoad(u2, p, 0).x;
    if (face_min(s1, axis, p, grid.dims) <= face_min(s2, axis, p, grid.dims)) {
        u = textureLoad(u1, p, 0).x;
    }
    textureStore(dst, p, vec4<f32>(u, 0.0, 0.0, 0.0));
}
```

- [ ] **Step 3: `collider.rs` and the union**

Create `crates/elements-ember/src/collider.rs`, following `shape_emitter.rs`'s structure:

```rust
//! `ember.collider`: a keyframed sphere or box that the fluid cannot enter
//! (2b-2 spec §2.3). It outputs a signed-distance field and the collider's
//! velocity at every face, the pair piece 4's mesh voxelizer will also produce.

use elements_core::gpu::{Axis, ComputeBatch, Field, GpuContext, GpuError, PipelineCache, StaggeredField};
use elements_core::graph::{DocError, EvalCtx, Node, NodeError, SocketSpec, SocketType, Value};
use serde::{Deserialize, Serialize};

use crate::kernels::{Bind, axis_index, bind_group, uniform_buffer};
use crate::node_util::produce;
use crate::params;
use crate::transform::{Pose, Shape, ShapeGpu, Transform};

pub const KIND: &str = "ember.collider";

const CELLS_WGSL: &str = concat!(
    include_str!("kernels/shaders/collider_common.wgsl"),
    include_str!("kernels/shaders/shape.wgsl"),
    include_str!("kernels/shaders/collider_cells.wgsl"),
);

const FACES_WGSL: &str = concat!(
    include_str!("kernels/shaders/collider_common.wgsl"),
    include_str!("kernels/shaders/shape.wgsl"),
    include_str!("kernels/shaders/collider_faces.wgsl"),
);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColliderParams {
    pub shape: Shape,
    pub transform: Transform,
}

/// A collider's two outputs.
#[derive(Clone, Copy)]
pub struct ColliderFields<'a> {
    /// Signed distance, metres, negative inside.
    pub sdf: &'a Field,
    /// The collider's velocity at every face, m/s.
    pub velocity: &'a StaggeredField,
}

impl ColliderFields<'_> {
    pub(crate) fn check(&self, what: &str) -> Result<(), GpuError> {
        if self.velocity.cells() != self.sdf.dims() {
            return Err(GpuError::Validation(format!(
                "{what}: velocity {:?}, sdf {:?}",
                self.velocity.cells(),
                self.sdf.dims()
            )));
        }
        Ok(())
    }
}

/// Matches `Collider` in collider_common.wgsl, 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ColliderGpu {
    dims: [u32; 3],
    dx: f32,
    axis: u32,
    _pad: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<ColliderGpu>() == 32);

/// Write `params`' SDF and velocity at `pose` into `out`. Submits its own batch.
pub fn fill_collider(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    params: &ColliderParams,
    pose: &Pose,
    dx: f32,
    out: ColliderFields<'_>,
) -> Result<(), GpuError> {
    out.check("fill_collider")?;
    let cells = out.sdf.dims();
    let dims = [cells.x, cells.y, cells.z];
    let cell_pipe = cache.get_or_create(gpu, "ember.collider.cells", CELLS_WGSL, "main")?;
    let face_pipe = cache.get_or_create(gpu, "ember.collider.faces", FACES_WGSL, "main")?;
    let shape = uniform_buffer(gpu, "ember-shape", bytemuck::bytes_of(&ShapeGpu::new(&params.shape, pose)))?;
    let make = |axis: u32| ColliderGpu { dims, dx, axis, _pad: [0; 3] };
    let mut batch = ComputeBatch::new();
    let cell_params = uniform_buffer(gpu, "ember-collider", bytemuck::bytes_of(&make(0)))?;
    let group = bind_group(gpu, &cell_pipe, &[Bind::Tex(out.sdf), Bind::Buf(&cell_params), Bind::Buf(&shape)])?;
    batch.dispatch(&cell_pipe, &group, cells);
    for axis in Axis::ALL {
        let face_params = uniform_buffer(gpu, "ember-collider", bytemuck::bytes_of(&make(axis_index(axis))))?;
        let face = out.velocity.face(axis);
        let group = bind_group(gpu, &face_pipe, &[Bind::Tex(face), Bind::Buf(&face_params), Bind::Buf(&shape)])?;
        batch.dispatch(&face_pipe, &group, face.dims());
    }
    batch.submit(gpu)
}

#[derive(Debug, Clone)]
pub struct Collider {
    params: ColliderParams,
}

impl Node for Collider {
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
            fill_collider(gpu, cache, params, &pose, dx, ColliderFields { sdf: &cells[0], velocity })
        })
    }
}

pub(crate) fn build(params: &serde_json::Value) -> Result<Box<dyn Node>, DocError> {
    let p: ColliderParams = params::parse(KIND, params)?;
    p.shape.validate(KIND)?;
    p.transform.validate(KIND)?;
    Ok(Box::new(Collider { params: p }))
}
```

In `unions.rs`, add `ember.collider_union`, mirroring the emitter union:

```rust
pub const COLLIDER_UNION_KIND: &str = "ember.collider_union";

const COLLIDER_CELLS_WGSL: &str = include_str!("kernels/shaders/collider_union_cells.wgsl");
const COLLIDER_FACES_WGSL: &str = concat!(
    include_str!("kernels/shaders/weights.wgsl"),
    include_str!("kernels/shaders/collider_union_faces.wgsl"),
);

/// `out` = the union of colliders `a` and `b`. Submits its own batch.
pub fn union_colliders(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    a: ColliderFields<'_>,
    b: ColliderFields<'_>,
    out: ColliderFields<'_>,
) -> Result<(), GpuError> {
    a.check("union_colliders a")?;
    b.check("union_colliders b")?;
    out.check("union_colliders out")?;
    let cells = out.sdf.dims();
    if a.sdf.dims() != cells || b.sdf.dims() != cells {
        return Err(GpuError::Validation("union_colliders: inputs differ in size".to_owned()));
    }
    let dims = [cells.x, cells.y, cells.z];
    let cell_pipe = cache.get_or_create(gpu, "ember.collider_union.cells", COLLIDER_CELLS_WGSL, "main")?;
    let face_pipe = cache.get_or_create(gpu, "ember.collider_union.faces", COLLIDER_FACES_WGSL, "main")?;
    let grid = uniform_buffer(gpu, "ember-union", bytemuck::bytes_of(&GridGpu { dims, axis: 0 }))?;
    let mut batch = ComputeBatch::new();
    let group = bind_group(
        gpu,
        &cell_pipe,
        &[Bind::Tex(a.sdf), Bind::Tex(b.sdf), Bind::Tex(out.sdf), Bind::Buf(&grid)],
    )?;
    batch.dispatch(&cell_pipe, &group, cells);
    for axis in Axis::ALL {
        let grid = uniform_buffer(gpu, "ember-union", bytemuck::bytes_of(&GridGpu { dims, axis: axis_index(axis) }))?;
        let dst = out.velocity.face(axis);
        let group = bind_group(
            gpu,
            &face_pipe,
            &[
                Bind::Tex(a.sdf),
                Bind::Tex(b.sdf),
                Bind::Tex(a.velocity.face(axis)),
                Bind::Tex(b.velocity.face(axis)),
                Bind::Tex(dst),
                Bind::Buf(&grid),
            ],
        )?;
        batch.dispatch(&face_pipe, &group, dst.dims());
    }
    batch.submit(gpu)
}

#[derive(Debug, Clone)]
pub struct ColliderUnion;

impl Node for ColliderUnion {
    fn kind(&self) -> &'static str {
        COLLIDER_UNION_KIND
    }

    fn sockets(&self) -> SocketSpec {
        let group = [SocketType::Field, SocketType::VectorField];
        SocketSpec {
            inputs: [group, group].concat(),
            outputs: group.to_vec(),
        }
    }

    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let inputs = take_inputs(ctx, 4)?;
        let result = collider_union_node(ctx, &inputs);
        for value in inputs {
            ctx.release(value);
        }
        result
    }
}

fn collider_union_node(ctx: &mut EvalCtx<'_>, inputs: &[Value]) -> Result<Vec<Value>, NodeError> {
    let a = ColliderFields { sdf: inputs[0].as_field()?, velocity: inputs[1].as_vector_field()? };
    let b = ColliderFields { sdf: inputs[2].as_field()?, velocity: inputs[3].as_vector_field()? };
    produce(ctx, 1, |gpu, cache, cells, velocity| {
        union_colliders(gpu, cache, a, b, ColliderFields { sdf: &cells[0], velocity })
    })
}

pub(crate) fn build_collider_union(params: &serde_json::Value) -> Result<Box<dyn Node>, DocError> {
    if !(params.is_null() || params.as_object().is_some_and(|o| o.is_empty())) {
        return Err(crate::params::bad(COLLIDER_UNION_KIND, "takes no parameters"));
    }
    Ok(Box::new(ColliderUnion))
}
```

(`use crate::collider::ColliderFields;` in `unions.rs`.) Register both kinds in `lib.rs`, and add `pub mod collider;`.

- [ ] **Step 4: Run and prove**

Run: `cargo nextest run -p elements-ember --test collider`
Expected: 4 passed.

| Test | Mutation | Expected |
|---|---|---|
| `the_sdf_is_the_signed_distance_to_the_shape` | in `shape_sdf`, drop `+ min(max(q.x, max(q.y, q.z)), 0.0)` | FAIL "box" (inside cells read 0) |
| `the_velocity_is_the_colliders_material_velocity_at_each_face` | in `collider_faces.wgsl`, delete `offset[axis] = 0.0;` (sample at cell centres) | FAIL on a face |
| `a_collider_union_takes_the_nearer_colliders_distance_and_velocity` | `<=` becomes `>` in `collider_union_faces.wgsl` | FAIL on a face |
| `bad_collider_parameters_are_rejected` | delete `p.shape.validate(KIND)?` in `collider::build` | FAIL |

- [ ] **Step 5: Run the gate and commit**

Run: `just check`
Expected: PASS.

```bash
git add crates/elements-ember
git commit   # subject: "Add keyframed sphere and box colliders, and collider unions"
```

The body explains why: a collider is a signed-distance field plus the collider's velocity at every face. That is the same pair a mesh voxelizer will produce, so the solver needs only one collider representation.

Include the mutation outputs.

---
## Task 5: Velocity emission and wind in the solver

Spec §3.1 (inputs 2 and 3), §3.3 and §3.4. The spec's task order put the solid mask first. Here emission comes first, so the solver's inputs grow from 2 to 4 to 6 and no task declares an input it ignores.

**Files:**
- Create: `crates/elements-ember/src/kernels/shaders/blend.wgsl`, `wind.wgsl`
- Modify: `crates/elements-ember/src/kernels/shaders/common.wgsl` and `crates/elements-ember/src/kernels/mod.rs` (`face_accel`, `StepConstants::wind`, `Uniforms::wind`)
- Modify: `crates/elements-ember/src/kernels/forces.rs` (`blend_velocity`, `wind`), `crates/elements-ember/src/node_util.rs` (`pair`, `take_listed`)
- Modify: `crates/elements-ember/src/solver.rs` (`Emission`, `Sources`, four inputs, the `wind` parameter)
- Modify: every `Sources { density, temperature }` literal (tests/solver.rs:124 and :499, tests/scenes.rs:62, :137, :197 and :257, examples/speed_gate.rs:115) → `Sources::new(density, temperature)`
- Test: `crates/elements-ember/tests/forces.rs`, `crates/elements-ember/tests/solver.rs`

**Interfaces:**
- Consumes: `face_weight` in `weights.wgsl` (Task 2), `EvalCtx::input_connected` and `NodeError::IncompletePair` (Task 1), and `ember.emitter`'s four outputs (Task 2).
- Produces:
  - `StepConstants::wind: [f32; 3]` (m/s², default zero) and `Uniforms::wind(&self) -> [f32; 3]`.
  - `kernels::blend_velocity(gpu, cache, batch, u, velocity: &StaggeredField, weight: &Field, target: &StaggeredField) -> Result<(), GpuError>`
  - `kernels::wind(gpu, cache, batch, u, velocity: &StaggeredField) -> Result<(), GpuError>`
  - `solver::Emission<'a> { weight: &'a Field, velocity: &'a StaggeredField }` (Copy).
  - `Sources<'a> { density, temperature, emission: Option<Emission<'a>> }`, with `Sources::new(density, temperature)` and `Sources::with_emission(self, e: Emission<'a>) -> Self`.
  - `SolverParams::wind: [f32; 3]`, and a `wind` document key.
  - `node_util::pair(ctx: &EvalCtx<'_>, a: u32, b: u32) -> Result<bool, NodeError>` and `node_util::take_listed(ctx: &mut EvalCtx<'_>, indices: &[u32]) -> Result<Vec<(u32, Value)>, NodeError>`
- Solver sockets: `[Field density rate, Field temperature rate, Field velocity weight, VectorField target velocity]`. Inputs 2 and 3 are optional as a pair.

- [ ] **Step 1: Write the failing tests**

In `tests/forces.rs` (add `blend_velocity`, `wind` to the kernels import):

```rust
/// Spec §3.3: where the weight is w, still fluid approaches the target as
/// 1 − exp(−w·t), independent of the substep. Wall faces are untouched.
#[test]
fn velocity_emission_approaches_the_target_exponentially() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let c = StepConstants::new(CELLS, 0.1, 0.125);
    let zero: [Vec<f32>; 3] = std::array::from_fn(|a| vec![0.0; face_dims(CELLS, a).voxel_count()]);
    let velocity = upload_staggered(&gpu, &mut pool, CELLS, &zero);
    let weight = upload(&gpu, &mut pool, CELLS, &vec![2.0; CELLS.voxel_count()]);
    let target_faces: [Vec<f32>; 3] = std::array::from_fn(|a| {
        vec![if a == 0 { 1.0 } else { 0.0 }; face_dims(CELLS, a).voxel_count()]
    });
    let target = upload_staggered(&gpu, &mut pool, CELLS, &target_faces);
    let u = Uniforms::new(&gpu, &c).unwrap();
    for _ in 0..3 {
        let mut batch = ComputeBatch::new();
        blend_velocity(&gpu, &mut cache, &mut batch, &u, &velocity, &weight, &target).unwrap();
        batch.submit(&gpu).unwrap();
    }
    let got = read_staggered(&gpu, &velocity);
    let want = 1.0 - (-2.0f32 * 0.3).exp();
    let d = face_dims(CELLS, 0);
    for k in 0..d.z {
        for j in 0..d.y {
            for i in 0..d.x {
                let v = got[0][index(d, i, j, k)];
                if is_wall(CELLS, 0, i) {
                    assert_eq!(v, 0.0, "wall x face {i}");
                } else {
                    assert!((v - want).abs() <= 1e-5, "x face {:?}: {v} vs {want}", [i, j, k]);
                }
            }
        }
    }
    assert!(got[1].iter().chain(&got[2]).all(|&v| v == 0.0));
}

/// Spec §3.4: wind is a uniform acceleration on every non-wall face.
#[test]
fn wind_accelerates_every_non_wall_face_uniformly() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let c = StepConstants {
        wind: [0.5, 0.0, -1.0],
        ..StepConstants::new(CELLS, 0.1, 0.125)
    };
    let zero: [Vec<f32>; 3] = std::array::from_fn(|a| vec![0.0; face_dims(CELLS, a).voxel_count()]);
    let velocity = upload_staggered(&gpu, &mut pool, CELLS, &zero);
    let u = Uniforms::new(&gpu, &c).unwrap();
    for _ in 0..4 {
        let mut batch = ComputeBatch::new();
        wind(&gpu, &mut cache, &mut batch, &u, &velocity).unwrap();
        batch.submit(&gpu).unwrap();
    }
    let got = read_staggered(&gpu, &velocity);
    for (a, rate) in [(0usize, 0.5f32), (1, 0.0), (2, -1.0)] {
        let d = face_dims(CELLS, a);
        for k in 0..d.z {
            for j in 0..d.y {
                for i in 0..d.x {
                    let want = if is_wall(CELLS, a, [i, j, k][a]) { 0.0 } else { 4.0 * 0.1 * rate };
                    let v = got[a][index(d, i, j, k)];
                    assert!((v - want).abs() <= 1e-6, "face {a} {:?}: {v} vs {want}", [i, j, k]);
                }
            }
        }
    }
}
```

In `tests/solver.rs`:

```rust
/// Spec §3.1: velocity emission needs its weight and its target together.
#[test]
fn a_velocity_weight_without_a_target_is_an_error() {
    let registry = elements_ember::registry();
    let build = |with_target: bool| {
        let mut graph = Graph::new();
        let emitter = graph.add_node(
            registry
                .build(
                    elements_ember::shape_emitter::KIND,
                    &serde_json::json!({
                        "shape": { "box": { "half_extents": [0.2, 0.2, 0.2] } },
                        "transform": { "keys": [{ "frame": 0, "translate": [1.0, 1.0, 0.4] }] },
                        "density_rate": 1.0, "velocity": [0.0, 0.0, 1.0], "velocity_blend": 10.0
                    }),
                )
                .unwrap(),
        );
        let solver = graph.add_node(registry.build(KIND, &serde_json::json!({ "pressure_iterations": 4 })).unwrap());
        let output = graph.add_node(registry.build("core.output", &serde_json::json!({})).unwrap());
        let socket = |node: NodeId, index: u32| SocketId { node, index };
        let last = if with_target { 4 } else { 3 };
        for index in 0..last {
            graph.connect(socket(emitter, index), socket(solver, index)).unwrap();
        }
        graph.connect(socket(solver, 0), socket(output, 0)).unwrap();
        graph.set_output(output);
        graph
    };
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let run = |graph: &Graph, pool: &mut FieldPool, pipelines: &mut PipelineCache| {
        let mut state = StateStore::new();
        let result = graph.eval_frame(&gpu, pool, pipelines, &mut state, Time::at(1, 1, 24.0), FieldDims::new(8, 8, 8));
        state.clear(pool);
        result.map(|evaluated| evaluated.value.release_to(pool))
    };
    let err = run(&build(false), &mut pool, &mut pipelines).unwrap_err();
    assert!(
        matches!(err, NodeError::IncompletePair { connected: 2, missing: 3, .. }),
        "got {err:?}"
    );
    run(&build(true), &mut pool, &mut pipelines).unwrap();
}
```

Extend `step_constants_carry_every_solver_parameter` with `"wind": [0.5, 0.0, -1.0]` in its document and `assert_eq!(c.wind, [0.5, 0.0, -1.0]);`. Add `assert!(rejected(serde_json::json!({ "wind": [1e39, 0.0, 0.0] })));` to the parameter test.

Run: `cargo nextest run -p elements-ember --test forces --test solver`
Expected: compile errors: `blend_velocity`, `wind`, `StepConstants::wind` and `IncompletePair`, used by solver code.

- [ ] **Step 2: The uniform and the kernels**

In `common.wgsl`'s `Params`, `_pad0: u32` becomes:

```wgsl
    face_accel: f32,     // wind along this uniform's axis, m/s²; 0 for cell grids
```

In `kernels/mod.rs`: in `KernelParams`, `_pad: [u32; 3]` becomes `face_accel: f32, _pad: [u32; 2]` (still 64 bytes). `StepConstants` gains

```rust
    /// Wind, a uniform acceleration, m/s² (spec §3.4).
    pub wind: [f32; 3],
```

with `wind: [0.0; 3]` in `new`. `Uniforms` stores `wind: [f32; 3]` and exposes `pub(crate) fn wind(&self) -> [f32; 3]`. `make(axis, decay)` fills `face_accel: if axis < 3 { c.wind[axis as usize] } else { 0.0 }` and `_pad: [0; 2]`.

Create `crates/elements-ember/src/kernels/shaders/blend.wgsl`:

```wgsl
// Velocity emission (2b-2 spec §3.3): u += (u_e − u)·(1 − exp(−w·h)) on one
// axis's faces, where w is the face's velocity weight in 1/s. The exponential
// makes the result independent of the substep length. Wall faces are left alone.

@group(0) @binding(0) var face: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var weight: texture_3d<f32>;
@group(0) @binding(2) var goal: texture_3d<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let axis = params.axis;
    if (any(gid >= grid_dims(axis))) {
        return;
    }
    if (is_wall(axis, gid[axis])) {
        return;
    }
    let p = vec3<i32>(gid);
    let w = face_weight(weight, axis, p, params.dims);
    if (w <= 0.0) {
        return;
    }
    let u = textureLoad(face, p).x;
    let next = u + (textureLoad(goal, p, 0).x - u) * (1.0 - exp(-w * params.h));
    textureStore(face, p, vec4<f32>(next, 0.0, 0.0, 0.0));
}
```

Create `crates/elements-ember/src/kernels/shaders/wind.wgsl`:

```wgsl
// Wind (2b-2 spec §3.4): u += h·a on one axis's faces, where a is the wind
// along that axis. Wall faces stay as they are.

@group(0) @binding(0) var face: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let axis = params.axis;
    if (any(gid >= grid_dims(axis))) {
        return;
    }
    if (is_wall(axis, gid[axis])) {
        return;
    }
    let p = vec3<i32>(gid);
    let u = textureLoad(face, p).x + params.h * params.face_accel;
    textureStore(face, p, vec4<f32>(u, 0.0, 0.0, 0.0));
}
```

In `kernels/forces.rs`, add:

```rust
const BLEND: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/weights.wgsl"),
    include_str!("shaders/blend.wgsl"),
);

const WIND: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/wind.wgsl"));

/// Pull every non-wall face toward `target` by 1 − exp(−w·h), where w is the
/// face's velocity weight (spec §3.3).
pub fn blend_velocity(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    velocity: &StaggeredField,
    weight: &Field,
    target: &StaggeredField,
) -> Result<(), GpuError> {
    let cells = u.cells();
    expect_dims("blend weight", weight, cells)?;
    if velocity.cells() != cells || target.cells() != cells {
        return Err(GpuError::Validation(format!(
            "blend_velocity: velocity {:?}, target {:?}, domain {cells:?}",
            velocity.cells(),
            target.cells()
        )));
    }
    let pipeline = cache.get_or_create(gpu, "ember.blend", BLEND, "main")?;
    for axis in Axis::ALL {
        let face = velocity.face(axis);
        let group = bind_group(
            gpu,
            &pipeline,
            &[Bind::Tex(face), Bind::Tex(weight), Bind::Tex(target.face(axis)), Bind::Buf(u.axis(axis))],
        )?;
        batch.dispatch(&pipeline, &group, face.dims());
    }
    Ok(())
}

/// Add h·wind at every non-wall face, on each axis whose wind is nonzero (spec §3.4).
pub fn wind(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    velocity: &StaggeredField,
) -> Result<(), GpuError> {
    if velocity.cells() != u.cells() {
        return Err(GpuError::Validation(format!(
            "wind: velocity {:?}, domain {:?}",
            velocity.cells(),
            u.cells()
        )));
    }
    let pipeline = cache.get_or_create(gpu, "ember.wind", WIND, "main")?;
    for (axis, a) in Axis::ALL.into_iter().zip(u.wind()) {
        if a == 0.0 {
            continue;
        }
        let face = velocity.face(axis);
        let group = bind_group(gpu, &pipeline, &[Bind::Tex(face), Bind::Buf(u.axis(axis))])?;
        batch.dispatch(&pipeline, &group, face.dims());
    }
    Ok(())
}
```

Export both from `kernels` (`pub use forces::{blend_velocity, buoyancy, emit, wind};`).

- [ ] **Step 3: The solver**

In `node_util.rs`, add:

```rust
/// Whether inputs `a` and `b`, which mean something only together, are
/// connected: both (true), neither (false), or one alone (an error naming both).
pub(crate) fn pair(ctx: &EvalCtx<'_>, a: u32, b: u32) -> Result<bool, NodeError> {
    let node = ctx.node_id();
    match (ctx.input_connected(a), ctx.input_connected(b)) {
        (true, true) => Ok(true),
        (false, false) => Ok(false),
        (true, false) => Err(NodeError::IncompletePair { node, connected: a, missing: b }),
        (false, true) => Err(NodeError::IncompletePair { node, connected: b, missing: a }),
    }
}

/// Take the listed inputs, in order, with their indices. On failure, every
/// input already taken is released.
pub(crate) fn take_listed(ctx: &mut EvalCtx<'_>, indices: &[u32]) -> Result<Vec<(u32, Value)>, NodeError> {
    let mut taken = Vec::with_capacity(indices.len());
    for &i in indices {
        match ctx.take_input(i) {
            Ok(value) => taken.push((i, value)),
            Err(e) => {
                for (_, value) in taken {
                    ctx.release(value);
                }
                return Err(e);
            }
        }
    }
    Ok(taken)
}
```

In `solver.rs`:

1. Add, next to `Sources`:

```rust
/// Velocity emission for one frame: the velocity weight (1/s) and the
/// target velocity (spec §3.3).
#[derive(Clone, Copy)]
pub struct Emission<'a> {
    pub weight: &'a Field,
    pub velocity: &'a StaggeredField,
}
```

and replace `Sources` with:

```rust
/// A frame's inputs to the solver.
#[derive(Clone, Copy)]
pub struct Sources<'a> {
    /// Emission rates per second, at the domain's dims.
    pub density: &'a Field,
    pub temperature: &'a Field,
    /// Velocity emission, when an emitter's weight and target are connected.
    pub emission: Option<Emission<'a>>,
}

impl<'a> Sources<'a> {
    pub fn new(density: &'a Field, temperature: &'a Field) -> Self {
        Self { density, temperature, emission: None }
    }

    pub fn with_emission(self, emission: Emission<'a>) -> Self {
        Self { emission: Some(emission), ..self }
    }
}
```

Migrate every `Sources { density: X, temperature: Y }` literal to `Sources::new(X, Y)`. They are at tests/solver.rs:124 and :499, tests/scenes.rs:62, :137, :197 and :257, and examples/speed_gate.rs:115; `cargo build --all-targets` lists any others.

2. `Substep` gains `wind: bool`, which is `constants.wind != [0.0; 3]` in `Substep::new`. In `pre_projection`, after the two `emit` calls:

```rust
        if let Some(e) = sources.emission {
            kernels::blend_velocity(gpu, cache, &mut self.batch, u, &state.velocity, e.weight, e.velocity)?;
        }
```

and after `buoyancy`:

```rust
        if self.wind {
            kernels::wind(gpu, cache, &mut self.batch, u, &state.velocity)?;
        }
```

3. `SolverParams` gains `pub wind: [f32; 3]` (doc: `/// Wind, a uniform acceleration, m/s² (spec §3.4).`). `Quality::params` sets `wind: [0.0; 3]`, and `DocParams` gains `wind: Option<[f32; 3]>`, resolved with `unwrap_or(preset.wind)`. `validate` adds `params::finite(KIND, "wind", &p.wind)?;`, and `step_constants` passes `wind: self.wind`.

4. Sockets: `inputs: vec![SocketType::Field, SocketType::Field, SocketType::Field, SocketType::VectorField]`. Replace `run`'s input handling:

```rust
    fn run(&self, ctx: &mut EvalCtx<'_>, state: &mut SolverState) -> Result<Vec<Value>, NodeError> {
        let mut wanted = vec![0, 1];
        if pair(ctx, 2, 3)? {
            wanted.extend([2, 3]);
        }
        let inputs = take_listed(ctx, &wanted)?;
        let stepped = self.step(ctx, state, &inputs);
        for (_, value) in inputs {
            ctx.release(value);
        }
        stepped?;

        // The outputs are copies: the state stays in the store for the next
        // frame. Outputs nobody reads are not copied at all.
        let wanted: [bool; 3] = std::array::from_fn(|i| ctx.output_wanted(i as u32));
        ctx.with_gpu_pool(|gpu, _, pool| copy_outputs(gpu, pool, state, wanted))
    }
```

`step` takes `inputs: &[(u32, Value)]` in place of the two `&Value`s and builds `Sources` from them:

```rust
        let node = ctx.node_id();
        let find = |index: u32| inputs.iter().find(|(i, _)| *i == index).map(|(_, v)| v);
        let field = |index: u32| -> Result<&Field, NodeError> {
            find(index)
                .ok_or(NodeError::MissingInput { node, index })?
                .as_field()
                .map_err(|_| NodeError::TypeMismatch { node, index, expected: SocketType::Field })
        };
        let vector = |index: u32| -> Result<&StaggeredField, NodeError> {
            find(index)
                .ok_or(NodeError::MissingInput { node, index })?
                .as_vector_field()
                .map_err(|_| NodeError::TypeMismatch { node, index, expected: SocketType::VectorField })
        };
        let mut sources = Sources::new(field(0)?, field(1)?);
        if find(2).is_some() {
            sources = sources.with_emission(Emission { weight: field(2)?, velocity: vector(3)? });
        }
```

If the closures' borrowed return types do not infer, write `field` and `vector` as small free functions taking `inputs` and `node`.

- [ ] **Step 4: Run and prove**

Run: `cargo nextest run -p elements-ember`
Expected: all pass. Existing two-input documents step unchanged.

| Test | Mutation | Expected |
|---|---|---|
| `velocity_emission_approaches_the_target_exponentially` | in `blend.wgsl`, `(1.0 - exp(-w * params.h))` becomes `(w * params.h)` | FAIL: 0.488 vs 0.451 |
| `wind_accelerates_every_non_wall_face_uniformly` | delete the `is_wall` early return in `wind.wgsl` | FAIL on a wall face |
| `a_velocity_weight_without_a_target_is_an_error` | `pair` returns `Ok(false)` for `(true, false)` | FAIL: no error |
| `step_constants_carry_every_solver_parameter` | drop `wind: self.wind` in `step_constants` | FAIL: wind |

- [ ] **Step 5: Run the gate and commit**

Run: `just check`
Expected: PASS.

```bash
git add crates/elements-ember
git commit   # subject: "Emit velocity toward a target, and add wind"
```

The body explains why:
- Emitters can now drive jets, and a scene can have wind.
- The blend is exponential in `h`, so jets look the same whatever the substep count.
- The solver's inputs grow as an optional pair, so existing two-input documents are untouched.

Include the mutation outputs.

---

## Task 6: Solids in the solver

Spec §3.1 (inputs 4 and 5), §3.2 and §5's collider tests, with the second correction noted at the top: solid corners take the mean of the fluid corners.

**Files:**
- Create: `crates/elements-ember/src/kernels/shaders/solid.wgsl`, `solidify.wgsl`, `crates/elements-ember/src/kernels/solid.rs`
- Modify: `common.wgsl` and `kernels/mod.rs` (`has_solids`, `Solids`, `Bind::View`, the placeholder, `solid_views`)
- Modify: `shaders/advect.wgsl`, `maccormack.wgsl`, `gradient.wgsl`, `pressure.wgsl`, `buoyancy.wgsl`, `curl.wgsl`, `confine.wgsl`, `blend.wgsl`, `wind.wgsl`, and their Rust wrappers in `kernels/advect.rs`, `project.rs`, `forces.rs` and `vorticity.rs`
- Modify: `crates/elements-ember/src/solver.rs` (six inputs, the mask, `Sources::solids`)
- Modify: every test and example calling a changed kernel function (the compiler lists them: tests/advection.rs, projection.rs, forces.rs, vorticity.rs and scenes.rs) — pass `None` as the new last argument
- Modify: `docs/superpowers/specs/2026-09-23-ember-scene-content-2b2-design.md` (§3.2's scalar-sampling rule)
- Test: `tests/projection.rs`, `tests/advection.rs`, `tests/scenes.rs`, `tests/solver.rs`

**Interfaces:**
- Consumes: `fill_collider`, `ColliderParams`, `ColliderFields` (Task 4); `pair`, `take_listed`, `Sources` (Task 5).
- Produces:
  - `kernels::Solids<'a> { mask: &'a Field, velocity: &'a StaggeredField }` (Copy)
  - `StepConstants::has_solids: bool` (default false) and `Uniforms::has_solids(&self) -> bool`
  - `kernels::solidify(gpu, cache, batch, u, sdf: &Field, mask: &Field) -> Result<(), GpuError>`
  - A new last parameter, `solids: Option<Solids<'_>>`, on `advect`, `maccormack`, `subtract_gradient`, `pressure`, `solve_pressure`, `buoyancy`, `curl`, `confine`, `blend_velocity` and `wind`.
  - `Sources::solids: Option<Solids<'a>>` and `Sources::with_solids(self, s: Solids<'a>) -> Self`
  - `Substep::project(…, iterations, solids: Option<Solids<'_>>)` and `Substep::advect_scalars(…, solids: Option<Solids<'_>>)`
  - WGSL `solid.wgsl`: `cell_solid(c)`, `face_solid(axis, p)`, `fluid_neighbour(c, d)`, `fluid_corners(tex, axis, p)`, `sample_fluid(tex, axis, p)`. Kernels including it must declare `var solid: texture_3d<f32>`.
- Solver sockets: `[F density, F temperature, F weight, V target, F collider sdf, V collider velocity]`. Inputs 4 and 5 are an optional pair.

- [ ] **Step 1: Write the failing tests**

Add a CPU helper to `tests/common/mod.rs`:

```rust
/// Mirrors `face_solid` in solid.wgsl: face `p` of `axis`'s grid touches a
/// solid cell (mask > 0.5) on either side.
pub fn face_solid_cpu(mask: &[f32], cells: FieldDims, axis: usize, p: [u32; 3]) -> bool {
    let n = [cells.x, cells.y, cells.z];
    let solid = |q: [i64; 3]| {
        (0..3).all(|a| q[a] >= 0 && q[a] < i64::from(n[a]))
            && mask[index(cells, q[0] as u32, q[1] as u32, q[2] as u32)] > 0.5
    };
    let mut below = p.map(i64::from);
    below[axis] -= 1;
    solid(below) || solid(p.map(i64::from))
}

/// A mask with a solid block of cells i in 4..7, j in 3..6 and k in 2..5.
pub fn block_mask(cells: FieldDims) -> Vec<f32> {
    let mut mask = vec![0.0; cells.voxel_count()];
    for k in 2..5 {
        for j in 3..6 {
            for i in 4..7 {
                mask[index(cells, i, j, k)] = 1.0;
            }
        }
    }
    mask
}
```

In `tests/projection.rs`:

```rust
/// Spec §3.2: after projection, every face touching a solid cell carries the
/// collider's velocity exactly, solid cells hold p = 0, and the fluid is
/// divergence-free. The input already carries the collider's velocity on
/// solid faces, as advection leaves it.
#[test]
fn projection_honours_solid_cells() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(12, 10, 8);
    let dx = 0.125;
    let c = StepConstants { has_solids: true, ..StepConstants::new(cells, 0.1, dx) };
    let mask_values = block_mask(cells);
    let obstacle_faces = velocity_pattern(cells);
    let mut faces = walled_velocity_pattern(cells);
    for a in 0..3 {
        let d = face_dims(cells, a);
        for k in 0..d.z {
            for j in 0..d.y {
                for i in 0..d.x {
                    if !is_wall(cells, a, [i, j, k][a]) && face_solid_cpu(&mask_values, cells, a, [i, j, k]) {
                        faces[a][index(d, i, j, k)] = obstacle_faces[a][index(d, i, j, k)];
                    }
                }
            }
        }
    }
    let mask = upload(&gpu, &mut pool, cells, &mask_values);
    let obstacle = upload_staggered(&gpu, &mut pool, cells, &obstacle_faces);
    let velocity = upload_staggered(&gpu, &mut pool, cells, &faces);
    let p = pool.acquire_zeroed(&gpu, &mut cache, cells).unwrap();
    let div = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    let u = Uniforms::new(&gpu, &c).unwrap();
    let solids = Some(Solids { mask: &mask, velocity: &obstacle });
    let mut batch = ComputeBatch::new();
    divergence(&gpu, &mut cache, &mut batch, &u, &velocity, &div).unwrap();
    solve_pressure(&gpu, &mut cache, &mut batch, &u, &p, &div, 200, solids).unwrap();
    subtract_gradient(&gpu, &mut cache, &mut batch, &u, &velocity, &p, solids).unwrap();
    batch.submit(&gpu).unwrap();

    let got = read_staggered(&gpu, &velocity);
    for a in 0..3 {
        let d = face_dims(cells, a);
        for k in 0..d.z {
            for j in 0..d.y {
                for i in 0..d.x {
                    if !is_wall(cells, a, [i, j, k][a]) && face_solid_cpu(&mask_values, cells, a, [i, j, k]) {
                        let at = index(d, i, j, k);
                        assert_eq!(got[a][at], obstacle_faces[a][at], "solid face {a} {:?}", [i, j, k]);
                    }
                }
            }
        }
    }
    let pv = p.read_back(&gpu).unwrap();
    for (n, m) in mask_values.iter().enumerate() {
        if *m > 0.5 {
            assert_eq!(pv[n], 0.0, "solid cell {n} holds p");
        }
    }
    // RMS divergence over fluid cells, before and after.
    let rms = |f: &[Vec<f32>; 3]| {
        let (xd, yd, zd) = (face_dims(cells, 0), face_dims(cells, 1), face_dims(cells, 2));
        let (mut sum, mut n) = (0.0f64, 0.0f64);
        for k in 0..cells.z {
            for j in 0..cells.y {
                for i in 0..cells.x {
                    if mask_values[index(cells, i, j, k)] > 0.5 {
                        continue;
                    }
                    let d = (f[0][index(xd, i + 1, j, k)] - f[0][index(xd, i, j, k)]
                        + f[1][index(yd, i, j + 1, k)] - f[1][index(yd, i, j, k)]
                        + f[2][index(zd, i, j, k + 1)] - f[2][index(zd, i, j, k)]) / dx;
                    sum += f64::from(d * d);
                    n += 1.0;
                }
            }
        }
        (sum / n).sqrt()
    };
    let (before, after) = (rms(&faces), rms(&got));
    assert!(after <= 0.1 * before, "fluid divergence {before} -> {after}");
}
```

In `tests/advection.rs`:

```rust
/// Spec §3.2 (as corrected): scalar sampling never reads a solid's contents.
/// Solid cells hold 100 and fluid cells 1. After a short advection step every
/// fluid cell still holds exactly 1, because solid corners are replaced by
/// the mean of the fluid corners.
#[test]
fn scalars_next_to_a_solid_never_sample_its_contents() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(12, 10, 8);
    // 0.6 m/s × 0.02 s / 0.125 m ≈ 0.1 cell, so every stencil keeps a fluid corner.
    let c = StepConstants {
        has_solids: true,
        open_mask: 0,
        ..StepConstants::new(cells, 0.02, 0.125)
    };
    let mask_values = block_mask(cells);
    let src_values: Vec<f32> = mask_values.iter().map(|&m| if m > 0.5 { 100.0 } else { 1.0 }).collect();
    let mask = upload(&gpu, &mut pool, cells, &mask_values);
    let obstacle = pool.acquire_staggered_zeroed(&gpu, &mut cache, cells).unwrap();
    let velocity = upload_staggered(&gpu, &mut pool, cells, &velocity_pattern(cells));
    let src = upload(&gpu, &mut pool, cells, &src_values);
    let dst = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    let u = Uniforms::new(&gpu, &c).unwrap();
    let mut batch = ComputeBatch::new();
    advect(
        &gpu, &mut cache, &mut batch, &u, Carried::Density, Pass::SemiLagrangian,
        &velocity, &src, &dst, Some(Solids { mask: &mask, velocity: &obstacle }),
    )
    .unwrap();
    batch.submit(&gpu).unwrap();
    let got = dst.read_back(&gpu).unwrap();
    for (n, m) in mask_values.iter().enumerate() {
        if *m < 0.5 {
            assert!((got[n] - 1.0).abs() <= 1e-5, "fluid cell {n}: {}", got[n]);
        }
    }
}
```

In `tests/scenes.rs` (import `elements_ember::collider::{ColliderFields, ColliderParams, fill_collider}`, `elements_ember::kernels::{Solids, solidify, Uniforms}`, `elements_ember::transform::{Key, Shape, Transform}`):

```rust
/// Build the frame's solid mask from `collider` at `frame`: returns (sdf values, mask, collider velocity).
fn collider_at(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    pool: &mut FieldPool,
    cells: FieldDims,
    dx: f32,
    collider: &ColliderParams,
    frame: f64,
) -> (Vec<f32>, Field, StaggeredField) {
    let sdf = pool.acquire(gpu, cells, FieldFormat::R32Float).unwrap();
    let velocity = pool.acquire_staggered_uninit(gpu, cells).unwrap();
    let pose = collider.transform.pose(frame, 1.0 / 24.0);
    fill_collider(gpu, cache, collider, &pose, dx, ColliderFields { sdf: &sdf, velocity: &velocity }).unwrap();
    let mask = pool.acquire(gpu, cells, FieldFormat::R32Float).unwrap();
    let u = Uniforms::new(gpu, &StepConstants::new(cells, 1.0, dx)).unwrap();
    let mut batch = ComputeBatch::new();
    solidify(gpu, cache, &mut batch, &u, &sdf, &mask).unwrap();
    batch.submit(gpu).unwrap();
    let values = sdf.read_back(gpu).unwrap();
    pool.release(sdf);
    (values, mask, velocity)
}

/// Umbrella §6: no density enters a collider. A plume rises into a static
/// sphere for 40 frames; the density inside stays at most 1% of the peak.
#[test]
fn a_plume_does_not_enter_a_collider() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(32, 32, 32);
    let dx = 2.0 / 32.0;
    let ds = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    let ts = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    let sphere = Sphere { center: [1.0, 1.0, 0.3], radius: 0.2, density_rate: 1.0, temperature_rate: 2.0 };
    fill_sphere(&gpu, &mut cache, &ds, &ts, &sphere, dx).unwrap();
    let collider = ColliderParams { shape: Shape::Sphere { radius: 0.25 }, transform: Transform::at([1.0, 1.0, 0.8]) };
    let (sdf, mask, obstacle) = collider_at(&gpu, &mut cache, &mut pool, cells, dx, &collider, 0.0);
    let sources = Sources::new(&ds, &ts).with_solids(Solids { mask: &mask, velocity: &obstacle });
    let constants = StepConstants { beta: 1.0, has_solids: true, ..StepConstants::new(cells, 1.0 / 24.0, dx) };
    let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, cells).unwrap();
    for _ in 0..40 {
        substep(&gpu, &mut cache, &mut pool, &mut state, sources, &constants, 160).unwrap();
    }
    let density = state.density.read_back(&gpu).unwrap();
    let peak = density.iter().copied().fold(0.0f32, f32::max);
    let inside = density.iter().zip(&sdf).filter(|(_, s)| **s < 0.0).map(|(d, _)| *d).fold(0.0f32, f32::max);
    assert!(peak > 0.0, "the plume must exist");
    assert!(inside <= 0.01 * peak, "inside {inside}, peak {peak}");
}

/// Spec §3.2: a moving collider pushes the fluid. A box sliding along +x at
/// 0.8 m/s through still air drives flow ahead of it in the same direction.
#[test]
fn a_moving_collider_pushes_the_fluid() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(24, 16, 16);
    let dx = 2.0 / 24.0;
    let zero = pool.acquire_zeroed(&gpu, &mut cache, cells).unwrap();
    let collider = ColliderParams {
        shape: Shape::Box { half_extents: [0.15; 3] },
        transform: Transform {
            keys: vec![
                Key { frame: 0.0, translate: [0.5, 0.67, 0.67], rotate: None },
                Key { frame: 24.0, translate: [1.3, 0.67, 0.67], rotate: None },
            ],
        },
    };
    let constants = StepConstants { has_solids: true, ..StepConstants::new(cells, 1.0 / 24.0, dx) };
    let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, cells).unwrap();
    let frames = 12;
    for frame in 0..frames {
        let (_, mask, obstacle) = collider_at(&gpu, &mut cache, &mut pool, cells, dx, &collider, f64::from(frame));
        let sources = Sources::new(&zero, &zero).with_solids(Solids { mask: &mask, velocity: &obstacle });
        substep(&gpu, &mut cache, &mut pool, &mut state, sources, &constants, 80).unwrap();
        pool.release(mask);
        pool.release_staggered(obstacle);
    }
    // The box's leading face, at the last frame stepped.
    let front = 0.5 + 0.8 * f64::from(frames - 1) / 24.0 + 0.15;
    let u = state.read_velocity(&gpu).unwrap();
    let d = face_dims(cells, 0);
    let (mut sum, mut n) = (0.0, 0.0);
    for k in 0..d.z {
        for j in 0..d.y {
            for i in 0..d.x {
                let x = f64::from(i) * f64::from(dx);
                let y = (f64::from(j) + 0.5) * f64::from(dx);
                let z = (f64::from(k) + 0.5) * f64::from(dx);
                if x > front + 0.5 * f64::from(dx) && x <= front + 2.5 * f64::from(dx)
                    && (y - 0.67).abs() <= 0.1 && (z - 0.67).abs() <= 0.1
                {
                    sum += f64::from(u[0][index(d, i, j, k)]);
                    n += 1.0;
                }
            }
        }
    }
    assert!(n > 0.0, "the slab ahead of the box must contain faces");
    assert!(sum / n > 0.1, "mean flow ahead of the box {}", sum / n);
}
```

In `tests/solver.rs`, add the following, using the file's existing `Session`, `timeline`, `plume_16` and `PREVIEW`:

```rust
/// `plume_16(solver)` plus an `ember.collider` wired to solver inputs 4 and
/// 5: a sphere of radius 0.1 at [10, 10, 10], wholly outside the domain.
/// With `velocity: false`, only the SDF (input 4) is wired.
fn plume_16_with_far_collider(solver: &str, velocity: bool) -> String {
    let mut doc: serde_json::Value = serde_json::from_str(&plume_16(solver)).unwrap();
    doc["nodes"].as_array_mut().unwrap().push(serde_json::json!({
        "id": 3, "kind": "ember.collider",
        "params": {
            "shape": { "sphere": { "radius": 0.1 } },
            "transform": { "keys": [{ "frame": 0, "translate": [10.0, 10.0, 10.0] }] }
        }
    }));
    let edges = doc["edges"].as_array_mut().unwrap();
    edges.push(serde_json::json!({ "from_node": 3, "from_index": 0, "to_node": 1, "to_index": 4 }));
    if velocity {
        edges.push(serde_json::json!({ "from_node": 3, "from_index": 1, "to_node": 1, "to_index": 5 }));
    }
    doc.to_string()
}

/// Spec §3.1: a collider entirely outside the domain changes nothing, bit for
/// bit, so the solid code paths are exact when nothing is solid.
#[test]
fn a_collider_outside_the_domain_changes_nothing() {
    let mut plain = Session::new(&plume_16(PREVIEW));
    let mut far = Session::new(&plume_16_with_far_collider(PREVIEW, true));
    let (mut a, mut b) = (timeline(0), timeline(0));
    for frame in 1..=10 {
        assert!(
            plain.density_bits(&mut a, frame) == far.density_bits(&mut b, frame),
            "frame {frame}"
        );
    }
}

/// Spec §3.1: a collider SDF without its velocity is an error naming both inputs.
#[test]
fn a_collider_sdf_without_its_velocity_is_an_error() {
    let text = plume_16_with_far_collider(PREVIEW, false);
    let (graph, dims) = Document::from_json(&text)
        .unwrap()
        .into_graph(&elements_ember::registry())
        .unwrap();
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let err = timeline(0)
        .goto(&graph, &gpu, &mut pool, &mut pipelines, dims, 1)
        .unwrap_err();
    assert!(
        matches!(err, NodeError::IncompletePair { connected: 4, missing: 5, .. }),
        "got {err:?}"
    );
}
```

`Session::new` takes the document's text. If its signature differs, adapt the call, not the assertions.

Run: `cargo nextest run -p elements-ember`
Expected: compile errors (`Solids`, `solidify`, `has_solids`, the new kernel arity).

- [ ] **Step 2: The shared solid module and the mask kernel**

In `common.wgsl`'s `Params`, `_pad1: u32` becomes:

```wgsl
    has_solids: u32,     // 1 when a collider's solid mask is bound (2b-2 spec §3.2)
```

In `kernels/mod.rs`, `KernelParams`' `_pad: [u32; 2]` becomes `has_solids: u32, _pad: u32` (still 64 bytes). `StepConstants` gains `pub has_solids: bool` (`false` in `new`, doc `/// Whether this substep's kernels read a solid mask (spec §3.2).`), and `make` fills `has_solids: u32::from(c.has_solids)`.

`Uniforms` gains three things:
- `has_solids: bool` and `pub(crate) fn has_solids(&self) -> bool`;
- a placeholder texture and view, created in `Uniforms::new`. Check `TextureDescriptor`'s fields against the vendored wgpu source:

```rust
        let placeholder = gpu.scoped(|| {
            gpu.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("ember-no-solids"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        })?;
        let placeholder_view = placeholder.create_view(&wgpu::TextureViewDescriptor::default());
```

- `pub(crate) fn placeholder(&self) -> &wgpu::TextureView`. Keep the `wgpu::Texture` in the struct too, so it outlives its view.

Add:

```rust
/// The frame's collider: the solid mask (1 inside) and the collider's
/// velocity at every face (spec §3.2).
#[derive(Clone, Copy)]
pub struct Solids<'a> {
    pub mask: &'a Field,
    pub velocity: &'a StaggeredField,
}

/// The `solid` and `obstacle` views a kernel binds: the mask and the
/// collider's velocity on `axis`'s faces, or the placeholder where a kernel
/// has no axis or there are no solids. The uniform's `has_solids` must agree
/// with `solids`, or kernels would read the placeholder as a mask.
pub(crate) fn solid_views<'a>(
    u: &'a Uniforms,
    solids: Option<Solids<'a>>,
    axis: Option<Axis>,
) -> Result<(&'a wgpu::TextureView, &'a wgpu::TextureView), GpuError> {
    if solids.is_some() != u.has_solids() {
        return Err(GpuError::Validation(
            "solids must be given exactly when StepConstants::has_solids is set".to_owned(),
        ));
    }
    let Some(s) = solids else {
        return Ok((u.placeholder(), u.placeholder()));
    };
    expect_dims("solid mask", s.mask, u.cells())?;
    if s.velocity.cells() != u.cells() {
        return Err(GpuError::Validation(format!(
            "collider velocity {:?}, domain {:?}",
            s.velocity.cells(),
            u.cells()
        )));
    }
    let obstacle = match axis {
        Some(a) => s.velocity.face(a).view(),
        None => u.placeholder(),
    };
    Ok((s.mask.view(), obstacle))
}
```

and a `Bind::View(&'a wgpu::TextureView)` variant, whose arm in `bind_group` is `Bind::View(view) => wgpu::BindingResource::TextureView(view)`.

Create `crates/elements-ember/src/kernels/shaders/solid.wgsl`:

```wgsl
// Collider solids (2b-2 spec §3.2), for kernels that declare
// `var solid: texture_3d<f32>`: the frame's cell mask, 1 inside a collider, or
// a 1×1×1 placeholder that is never read because params.has_solids is 0.

fn cell_solid(c: vec3<i32>) -> bool {
    if (params.has_solids == 0u) {
        return false;
    }
    if (any(c < vec3<i32>(0)) || any(c >= vec3<i32>(params.dims))) {
        return false;
    }
    return textureLoad(solid, c, 0).x > 0.5;
}

// Whether face `p` of `axis`'s grid touches a solid cell on either side.
fn face_solid(axis: u32, p: vec3<i32>) -> bool {
    if (params.has_solids == 0u) {
        return false;
    }
    var e = vec3<i32>(0);
    e[axis] = 1;
    return cell_solid(p - e) || cell_solid(p);
}

// Neighbour `c + d` for a difference stencil, or `c` itself when that
// neighbour is outside the domain or solid, so solids act like the domain edge.
fn fluid_neighbour(c: vec3<i32>, d: vec3<i32>) -> vec3<i32> {
    let n = c + d;
    if (any(n < vec3<i32>(0)) || any(n >= vec3<i32>(params.dims)) || cell_solid(n)) {
        return c;
    }
    return n;
}

// `corners()` for a cell-centred scalar near solids: each solid corner takes
// the mean of the fluid corners (0 when all eight are solid), so a solid's
// contents never reach the fluid through interpolation. Face grids and frames
// without solids are unchanged.
fn fluid_corners(tex: texture_3d<f32>, axis: u32, p: vec3<f32>) -> Corners {
    var k = corners(tex, axis, p);
    if (params.has_solids == 0u || axis != CELL) {
        return k;
    }
    let i0 = vec3<i32>(floor(clamp(p, vec3<f32>(-1.0), vec3<f32>(textureDimensions(tex, 0)))));
    var sum = 0.0;
    var count = 0.0;
    var solid_bits = 0u;
    for (var n = 0u; n < 8u; n = n + 1u) {
        let o = vec3<i32>(i32(n & 1u), i32((n >> 1u) & 1u), i32((n >> 2u) & 1u));
        if (cell_solid(i0 + o)) {
            solid_bits = solid_bits | (1u << n);
        } else {
            sum += k.c[n];
            count += 1.0;
        }
    }
    if (solid_bits == 0u) {
        return k;
    }
    let fill = select(0.0, sum / max(count, 1.0), count > 0.0);
    for (var n = 0u; n < 8u; n = n + 1u) {
        if (((solid_bits >> n) & 1u) == 1u) {
            k.c[n] = fill;
        }
    }
    return k;
}

// `sample_grid()` over `fluid_corners()`.
fn sample_fluid(tex: texture_3d<f32>, axis: u32, p: vec3<f32>) -> f32 {
    let k = fluid_corners(tex, axis, p);
    let c00 = mix(k.c[0], k.c[1], k.t.x);
    let c10 = mix(k.c[2], k.c[3], k.t.x);
    let c01 = mix(k.c[4], k.c[5], k.t.x);
    let c11 = mix(k.c[6], k.c[7], k.t.x);
    return mix(mix(c00, c10, k.t.y), mix(c01, c11, k.t.y), k.t.z);
}
```

Create `crates/elements-ember/src/kernels/shaders/solidify.wgsl`:

```wgsl
// The frame's solid mask (2b-2 spec §3.2): 1 where the collider's signed
// distance at the cell centre is below zero, 0 elsewhere.

@group(0) @binding(0) var sdf: texture_3d<f32>;
@group(0) @binding(1) var mask: texture_storage_3d<r32float, write>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    let solid = select(0.0, 1.0, textureLoad(sdf, p, 0).x < 0.0);
    textureStore(mask, p, vec4<f32>(solid, 0.0, 0.0, 0.0));
}
```

Create `crates/elements-ember/src/kernels/solid.rs`:

```rust
//! The frame's solid mask (2b-2 spec §3.2).

use elements_core::gpu::{ComputeBatch, Field, GpuContext, GpuError, PipelineCache};

use super::{Bind, Uniforms, bind_group, expect_dims};

const SOLIDIFY: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/solidify.wgsl"));

/// Record `mask` = 1 where `sdf` < 0, else 0.
pub fn solidify(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    u: &Uniforms,
    sdf: &Field,
    mask: &Field,
) -> Result<(), GpuError> {
    expect_dims("solidify sdf", sdf, u.cells())?;
    expect_dims("solidify mask", mask, u.cells())?;
    let pipeline = cache.get_or_create(gpu, "ember.solidify", SOLIDIFY, "main")?;
    let group = bind_group(gpu, &pipeline, &[Bind::Tex(sdf), Bind::Tex(mask), Bind::Buf(u.any())])?;
    batch.dispatch(&pipeline, &group, u.cells());
    Ok(())
}
```

In `kernels/mod.rs`, add `mod solid;` and `pub use solid::solidify;`.

- [ ] **Step 3: Solids in every boundary kernel**

Every shader below gains the listed bindings **after** its existing ones, so existing indices do not move. Each gets `include_str!("shaders/solid.wgsl")` concatenated right after `common.wgsl` in its Rust constant.

**`advect.wgsl`:** add `@group(0) @binding(6) var solid: texture_3d<f32>;` and `@group(0) @binding(7) var obstacle: texture_3d<f32>;`. In `pass_over`, replace the wall block and the sampling with:

```wgsl
    if (axis != CELL) {
        // Walls carry no normal velocity, and faces touching a collider carry
        // its velocity (spec §3.2), before projection as well as after, so the
        // divergence the solve sees matches what projection enforces.
        if (is_wall(axis, gid[axis])) {
            textureStore(dst, p, vec4<f32>(0.0));
            return;
        }
        if (face_solid(axis, p)) {
            textureStore(dst, p, vec4<f32>(textureLoad(obstacle, p, 0).x, 0.0, 0.0, 0.0));
            return;
        }
    }
    let x = vec3<f32>(gid) + grid_offset(axis);
    let value = sample_fluid(src, axis, backtrace(x, direction) - grid_offset(axis));
```

**`maccormack.wgsl`:** add bindings 8 (`solid`) and 9 (`obstacle`). Use the same two face checks, and `let k = fluid_corners(orig, axis, …)`.

**`gradient.wgsl`:** add bindings 3 (`solid`) and 4 (`obstacle`). After the wall check:

```wgsl
    if (face_solid(axis, p)) {
        textureStore(face, p, vec4<f32>(textureLoad(obstacle, p, 0).x, 0.0, 0.0, 0.0));
        return;
    }
```

**`pressure.wgsl`:** add binding 3 (`solid`). Right after the bounds and colour check:

```wgsl
    if (cell_solid(c)) {
        // A solid cell is not solved; fluid cells never read it (spec §3.2).
        textureStore(pressure, c, vec4<f32>(0.0));
        return;
    }
```

Each of the six inside-neighbour branches becomes, for example for −x:

```wgsl
    if (c.x > 0) {
        let q = c - vec3<i32>(1, 0, 0);
        if (!cell_solid(q)) {
            sum += textureLoad(pressure, q).x;
            count += 1.0;
        }
    } else if (is_open(0u, 0u)) {
        count += 1.0;
    }
```

Keep all six written out per side, as risk (j) requires. A solid neighbour is Neumann: left out, like a wall.

**`buoyancy.wgsl`:** add binding 4 (`solid`). After the boundary check: `if (face_solid(2u, above)) { return; }`.

**`curl.wgsl`:** add binding 8 (`solid`). The three differences become `centre_velocity(fluid_neighbour(c, vec3<i32>(1, 0, 0))) - centre_velocity(fluid_neighbour(c, vec3<i32>(-1, 0, 0)))`, and likewise for y and z.

**`confine.wgsl`:** add binding 6 (`solid`). `magnitude(c + …)` becomes `magnitude(fluid_neighbour(c, …))` in `force`. In `main`, compute `p` before the wall check and make it `if (is_wall(axis, i) || face_solid(axis, p)) { return; }`.

**`blend.wgsl`:** add binding 4 (`solid`) and `if (face_solid(axis, p)) { return; }` after the wall check (define `p` first).

**`wind.wgsl`:** add binding 2 (`solid`) and the same check.

In the Rust wrappers, append `solids: Option<Solids<'_>>` to each changed function. Get the views with `solid_views(u, solids, axis)`, where `axis` is `Some(axis)` for a face grid and `None` for a scalar or a kernel with no axis. Append `Bind::View(solid)`, plus `Bind::View(obstacle)` where the shader has `obstacle`, to each bind list. `solve_pressure` passes `solids` through to `pressure`. `Carried::Face(axis)` gives `Some(axis)`; `Density` and `Temperature` give `None`.

- [ ] **Step 4: The solver**

1. `Sources` gains `pub solids: Option<Solids<'a>>` (`None` in `new`) and `with_solids`.
2. `Substep` passes `sources.solids` to `buoyancy`, `blend_velocity`, `wind`, `confine_vorticity`'s `curl`/`confine`, and `advect_grid`'s `advect`/`maccormack`. `advect_grid` and `maccormack` gain a `solids` parameter. `project(…, iterations, solids)` passes it to `solve_pressure` and `subtract_gradient`, and `advect_scalars(…, solids)` to `advect_grid`. `substep()` passes `sources.solids` to both. Update the callers in tests/scenes.rs (lines 86 and 222) and examples/speed_gate.rs:132 with `None`.
3. Sockets: six inputs, `[F, F, F, V, F, V]`. In `run`, after the emission pair: `if pair(ctx, 4, 5)? { wanted.extend([4, 5]); }`.
4. In `step`, after building `sources`, when inputs 4 and 5 are present, build the mask before the CFL measurement and release it on every path:

```rust
        // Spec §3.2: the mask is rebuilt every frame from the collider's SDF,
        // before the CFL measurement, and never stored.
        let collider = match (find(4), find(5)) {
            (Some(_), Some(_)) => Some((field(4)?, vector(5)?)),
            _ => None,
        };
        let cells = ctx.dims();
        let dx = ctx.voxel_size();
        let mask = match collider {
            None => None,
            Some((sdf, _)) => Some(ctx.with_gpu_pool(|gpu, cache, pool| {
                let mask = pool.acquire(gpu, cells, FieldFormat::R32Float)?;
                let built = Uniforms::new(gpu, &StepConstants::new(cells, 1.0, dx)).and_then(|u| {
                    let mut batch = ComputeBatch::new();
                    kernels::solidify(gpu, cache, &mut batch, &u, sdf, &mask)?;
                    batch.submit(gpu)
                });
                match built {
                    Ok(()) => Ok(mask),
                    Err(e) => {
                        pool.release(mask);
                        Err(e)
                    }
                }
            })?),
        };
        if let (Some(mask), Some((_, velocity))) = (&mask, collider) {
            sources = sources.with_solids(Solids { mask, velocity });
        }
        let stepped = self.step_frame(ctx, state, sources);
        if let Some(mask) = mask {
            ctx.release(Value::Field(mask));
        }
        stepped
```

`step_frame` holds the old remainder of `step`: CFL, the constants and the substep loop. Its constants now also set `has_solids: sources.solids.is_some()`. With the mask created in one place and released in one place, the CFL error path cannot leak it.

- [ ] **Step 5: Correct the spec**

In `docs/superpowers/specs/2026-09-23-ember-scene-content-2b2-design.md` §3.2, replace the `texel()` bullet with:

"**Scalar sampling.** At an interpolation stencil touching a solid cell, each solid corner takes the mean of the fluid corners of the same eight, or 0 if all eight are solid. The spec first said 'clamp to the nearest fluid value along the axis'; that has no single meaning for a 3D stencil, so it was replaced while planning."

- [ ] **Step 6: Run and prove**

Run: `cargo nextest run -p elements-ember`
Expected: all pass. The existing tests pass `None` and `has_solids: false`, and must be unchanged.

If `a_plume_does_not_enter_a_collider` or `a_moving_collider_pushes_the_fluid` misses its threshold without a mutation, stop and report the measured values. Do not change a threshold.

| Test | Mutation | Expected |
|---|---|---|
| `projection_honours_solid_cells` | in `gradient.wgsl`, delete the `face_solid` branch | FAIL: solid face |
| `scalars_next_to_a_solid_never_sample_its_contents` | in `fluid_corners`, return `k` right after `corners()` | FAIL: a fluid cell near 100 |
| `a_plume_does_not_enter_a_collider` | in `solidify.wgsl`, always store 0.0 | FAIL: inside > 1% |
| `a_moving_collider_pushes_the_fluid` | in `collider_faces.wgsl`, store 0.0 | FAIL: mean ≈ 0 |
| `a_collider_outside_the_domain_changes_nothing` | in `cell_solid`, return `params.has_solids == 1u` (skip the mask read) | FAIL: bits differ |
| the pair test | `pair` returns `Ok(false)` for `(true, false)` | FAIL |

- [ ] **Step 7: Run the gate and commit**

Run: `just check`
Expected: PASS. Also time a MacCormack preview frame with no collider against before this task: `just bench-sweep 20,160`, then restore `docs/bench/iteration-sweep.md` with `git stash push -- docs/bench/iteration-sweep.md`. The slope should be unchanged within noise, because `has_solids` is a uniform branch. Report the numbers.

```bash
git add crates/elements-ember docs/superpowers/specs/2026-09-23-ember-scene-content-2b2-design.md
git commit   # subject: "Make colliders solid in every stage of the solver"
```

The body explains why:
- One mask, rebuilt every frame from the collider's SDF, reaches every place 2b-1 spec §4.2 listed.
- Solid faces carry the collider's velocity, so moving colliders push fluid.
- Scalar sampling was corrected, and the spec now says so.

Include the mutation outputs and the timing.

---

## Task 7: Animated determinism, the benchmark scenes, and docs

Spec §5 (determinism) and §6.

**Files:**
- Modify: `crates/elements-ember/tests/solver.rs` (a document-level frame-40 helper, and the animated test)
- Modify: `crates/elements-ember/src/bench.rs` (`Scene::collider`, `plume_collider`, `plume_wind`)
- Test: `crates/elements-ember/tests/bench.rs`
- Modify: `CLAUDE.md`, `README.md`, `docs/superpowers/specs/2026-09-21-ember-solver-design.md`, `docs/superpowers/specs/2026-09-23-ember-scene-content-2b2-design.md`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `bench::Scene::collider: Option<ColliderParams>`. `Scene` becomes `Clone` rather than `Copy`, because `Transform` holds a `Vec`.
  - `Scene::plume_collider(resolution) -> Scene` and `Scene::plume_wind(resolution) -> Scene`.

- [ ] **Step 1: Determinism under animation**

In `tests/solver.rs`, split `assert_frame_40_is_bit_identical(solver: &str)` into `assert_doc_frame_40_is_bit_identical(doc: &str)`, which holds the existing body and builds its graph from `doc`, and a one-line `assert_frame_40_is_bit_identical(solver)` that calls it with `plume_16(solver)`. Then add:

```rust
/// A 16³ document with an animated, noisy box emitter (all four outputs
/// wired) and an animated sphere collider (inputs 4 and 5).
const ANIMATED: &str = r#"{
  "version": 3, "dims": [16, 16, 16], "fps": 24.0, "domain_size": 2.0,
  "nodes": [
    { "id": 0, "kind": "ember.emitter", "params": {
        "shape": { "box": { "half_extents": [0.2, 0.2, 0.1] } },
        "transform": { "keys": [
          { "frame": 1, "translate": [0.7, 1.0, 0.3] },
          { "frame": 40, "translate": [1.3, 1.0, 0.3], "rotate": { "axis": [0, 0, 1], "degrees": 90 } } ] },
        "density_rate": 1.0, "temperature_rate": 2.0,
        "velocity": [0.0, 0.0, 0.5], "velocity_blend": 4.0,
        "noise": { "seed": 9, "scale_m": 0.2, "amplitude": 0.6, "evolution": 1.5 } } },
    { "id": 1, "kind": "ember.collider", "params": {
        "shape": { "sphere": { "radius": 0.2 } },
        "transform": { "keys": [
          { "frame": 1, "translate": [0.6, 1.0, 1.2] },
          { "frame": 40, "translate": [1.4, 1.0, 1.1] } ] } } },
    { "id": 2, "kind": "ember.smoke_solver", "params": { "pressure_iterations": 40, "buoyancy_temperature": 1.0, "vorticity": 2.0 } },
    { "id": 3, "kind": "core.output", "params": {} }
  ],
  "edges": [
    { "from_node": 0, "from_index": 0, "to_node": 2, "to_index": 0 },
    { "from_node": 0, "from_index": 1, "to_node": 2, "to_index": 1 },
    { "from_node": 0, "from_index": 2, "to_node": 2, "to_index": 2 },
    { "from_node": 0, "from_index": 3, "to_node": 2, "to_index": 3 },
    { "from_node": 1, "from_index": 0, "to_node": 2, "to_index": 4 },
    { "from_node": 1, "from_index": 1, "to_node": 2, "to_index": 5 },
    { "from_node": 2, "from_index": 0, "to_node": 3, "to_index": 0 }
  ],
  "output": 3
}"#;

/// Umbrella §4: frame 40 is bit-identical in order, after scrubbing and after
/// eviction, with an animated collider and an animated, noisy emitter,
/// because poses, masks and noise are recomputed from the document's time.
#[test]
fn frame_40_is_bit_identical_with_animated_emitter_and_collider() {
    assert_doc_frame_40_is_bit_identical(ANIMATED);
}
```

Run it. Expected: pass.

Mutation: in `ember.emitter`'s `eval`, cache the pose computed on the first frame seen, in a `std::sync::Mutex<Option<Pose>>` field of `Emitter`, and use it on every later frame. Expected: FAIL "after scrubbing". Record, restore.

- [ ] **Step 2: The benchmark scenes**

In `bench.rs`:
- `Scene` derives `Clone` (not `Copy`) and gains `pub collider: Option<ColliderParams>` (doc: `/// A static collider wired to the solver's inputs 4 and 5, if any.`). `plume` sets `collider: None`.
- Add:

```rust
    /// `plume` with a static sphere collider of radius 0.25 m, 0.5 m above the
    /// emitter (piece 2 spec §5.3).
    pub fn plume_collider(resolution: u32) -> Self {
        Self {
            name: "plume_collider",
            collider: Some(ColliderParams {
                shape: Shape::Sphere { radius: 0.25 },
                transform: Transform::at([1.0, 1.0, 0.8]),
            }),
            ..Self::plume(resolution)
        }
    }

    /// `plume` with wind of 0.5 m/s² along +x (piece 2 spec §5.3).
    pub fn plume_wind(resolution: u32) -> Self {
        let mut scene = Self::plume(resolution);
        scene.name = "plume_wind";
        scene.solver.wind = [0.5, 0.0, 0.0];
        scene
    }
```

- In `document()`, when `collider` is `Some`, add node `{ id: 3, kind: collider::KIND, params: serde_json::to_value(collider) }` and edges `(3, 0) → (1, 4)` and `(3, 1) → (1, 5)`.
- Fix any compile errors from `Scene` no longer being `Copy` in the examples: add `.clone()` where a scene is reused.

In `tests/bench.rs`:

Replace `the_plume_scene_loads_and_steps` with a helper and three tests:

```rust
/// The scene's document round-trips through JSON, builds, and emits by frame 2.
fn loads_and_steps(scene: &Scene) {
    let text = scene.document().to_json().unwrap();
    let doc = Document::from_json(&text).unwrap();
    let config = doc.timeline_config();
    let (graph, dims) = doc.into_graph(&elements_ember::registry()).unwrap();
    let gpu = common::gpu();
    let mut pool = FieldPool::new();
    let mut pipelines = PipelineCache::new();
    let mut timeline = Timeline::new(config);
    let frame = timeline
        .goto(&graph, &gpu, &mut pool, &mut pipelines, dims, 2)
        .unwrap();
    let density = frame.value.as_field().unwrap().read_back(&gpu).unwrap();
    assert!(density.iter().any(|&v| v > 0.0), "{} must emit", scene.name);
}

#[test]
fn the_plume_scene_loads_and_steps() {
    loads_and_steps(&Scene::plume(16));
}

#[test]
fn the_plume_collider_scene_loads_and_steps() {
    let scene = Scene::plume_collider(16);
    assert!(scene.document().nodes.iter().any(|n| n.kind == elements_ember::collider::KIND));
    loads_and_steps(&scene);
}

#[test]
fn the_plume_wind_scene_loads_and_steps() {
    let scene = Scene::plume_wind(16);
    assert_eq!(scene.solver.wind, [0.5, 0.0, 0.0]);
    loads_and_steps(&scene);
}
```

Mutation: in `document()`, wire the collider to inputs 4 and 4 (a duplicate edge). Expected: FAIL at graph build (`InputAlreadyConnected`) or `IncompletePair`. Record, restore.

- [ ] **Step 3: The docs**

- **`CLAUDE.md`, "What this is":** replace the 2b sentence's "**2b-2**, scene content (emitters, colliders, wind, flame), is next." with:

  "**2b-2**, scene content, is complete: keyframed box and sphere emitters (with noise and velocity emission) and colliders, unions of each, and wind (`docs/superpowers/specs/2026-09-23-ember-scene-content-2b2-design.md`). **2b-3**, the Mantaflow benchmark, is next. Flame is its own later cycle, 2b-4."

  Add the 2b-2 spec and plan to the document list. In "Constraints that bite", add: "**Kernels with solids** include `solid.wgsl` and must declare `var solid: texture_3d<f32>`. When there is no collider, a 1×1×1 placeholder is bound and `has_solids` = 0 keeps it unread."
- **Piece 2 spec status:** add "2b-2 (scene content) is complete; see `2026-09-23-ember-scene-content-2b2-design.md`."
- **2b-2 spec:** set `**Status:** Complete.`
- **`README.md`:** if its status or feature list describes emitters, colliders or wind as missing, update it to match. Check each claim against the code.

- [ ] **Step 4: Run the gate, commit and push**

Run: `just check`
Expected: PASS.

```bash
git add crates/elements-ember CLAUDE.md README.md docs
git commit   # subject: "Prove animated scenes are deterministic, and add the collider and wind benchmark scenes"
git push -u origin ember-scene-2b2
gh run list --branch ember-scene-2b2 --limit 1
```

The body explains why:
- Animation is the thing most likely to leak state across frames.
- 2b-3 needs the two scenes defined once, as data.

Wait for CI (`gh run watch <id>`), and report its conclusion. If it fails, report the failing test output; do not change tolerances.
