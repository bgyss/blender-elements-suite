# Elements Suite — Design

**Date:** 2026-09-19
**Status:** Approved for planning
**Scope of this document:** suite-level decomposition (normative), plus the full
specification for **Core v1**, which is the first implementation plan. Each
product (Ember, Tide, Strata, Weave) gets its own spec later.

> **This is the design as written before implementation, kept as written.**
> Several decisions changed once the code met reality, and the reasons are
> recorded in the plan's "Deviations from the spec" section rather than by
> editing this document into agreement with the outcome. Where the two differ,
> the plan and the code are authoritative. The substantive changes:
> the control plane is newline-delimited JSON, not CBOR; the data plane is a
> memory-mapped file, not POSIX shared memory; the add-on is pure Python with
> **no wheel**; scalar fields are `R32Float`, not `R16Float`, because the WebGPU
> baseline forbids `R16Float` as a storage texture; and section 3.6's data flow
> is not achieved (see the note there).

---

## 1. Goal

An open-source, Rust-based suite of real-time VFX authoring tools for Blender,
covering the same problem space as the JangaFX tools (volumetric gas, liquids,
terrain, procedural assets), using machine learning to make simulation easier
rather than only faster. Developed primarily by coding agents.

Naming is deliberately original. JangaFX product names are trademarks and are
used in this document only to identify prior art.

### Non-goals

- Matching JangaFX performance on CUDA-class hardware. Portability via `wgpu` is
  the accepted trade; expect 2-3x slower than a CUDA-native solver.
- Replacing Blender's Mantaflow for existing workflows.
- Shipping model weights inside the Blender extension.

---

## 2. Suite decomposition

One Rust workspace, one long-lived engine process, thin clients. This is the
*target* layout for the whole suite; Core v1 creates only `elements-core`,
`elements-io`, `elements-ipc`, `elementsd`, `elements-cli` and the addon.
Section 3.8 lists what Core v1 deliberately leaves out.

```
elements/                        (Apache-2.0 OR MIT)
  elements-core/       wgpu device, field pool, compute dispatch,
                       node graph, scheduler, time/cache
  elements-io/         OpenVDB write, Alembic, EXR sequences, flipbook atlases
  elements-ml/         inference seam (ONNX via `ort`), model registry
  elements-ipc/        shared-memory ring + control protocol
  elements-ember/      gas solver         -+
  elements-tide/       liquid solver       |  products, each depends on core
  elements-strata/     terrain             |
  elements-weave/      procedural assets  -+
  elementsd/           daemon binary
  elements-cli/        headless bake/render

blender-elements/                (GPL-3.0)
  __init__.py
  blender_manifest.toml
  wheels/              elements_ipc client, per platform
```

### 2.1 Architectural decisions

**D1 — Out-of-process daemon.** The engine is a separate process owning its own
`wgpu` device. The Blender addon is a client communicating over IPC.

*Rationale:* Blender's renderer is territorial about the GPU device; sharing a
context is fragile. Out-of-process gives crash isolation (a solver panic shows a
disconnect banner, not a lost `.blend`), lets the engine run headless in CI
without Blender, and keeps the permissively licensed engine free of any `bpy`
linkage.

**D2 — Node graph lives in core.** All four products are node graphs over
different data types. Core owns the graph, typed sockets, evaluation order and
serialization; products register node *kinds*.

*Rationale:* highest-leverage shared abstraction in the suite, and the one most
likely to be designed wrong. Core v1 therefore proves it with two trivial nodes
rather than fifty real ones.

**D3 — ML is a seam, never a dependency.** `elements-ml` exposes:

```rust
trait FieldTransform {
    fn apply(&self, input: &GpuField, output: &mut GpuField) -> Result<()>;
}
```

An ONNX model and a bicubic upsampler both implement it. Every product must
produce correct output with every model unloaded. Models ship as separately
downloadable packs.

**D4 — Split licensing.** Engine workspace is Apache-2.0 OR MIT; the Blender
addon is GPL-3.0. Consequence: **no GPL code may be copied into the engine.**
GPL prior art (FLIP Fluids, Blender internals, GPL-licensed research code) may
be read and its published algorithms reimplemented, but not vendored.

### 2.2 Build order

Core v1 -> Ember -> Tide -> Strata -> Weave.

Ember is second because grid-based gas exercises core's field and texture
machinery hardest. Tide reuses that and adds particles. Strata and Weave are
comparatively self-contained and may be parallelized once core is stable.

### 2.3 ML application per product

All of these are phase 2 within their product; none block that product's v1.

| Product | ML application | Prior art |
|---|---|---|
| Ember | Neural super-resolution at export (2-8x); learned sub-grid closure; video -> smoke VDB | RDN/GNN upres; WildSmoke; SmokeSVD |
| Tide | GNS surrogate for interactive preview; classical FLIP/MPM on bake | GNS (~5000x vs MPM) |
| Strata | Latent-diffusion heightmap + texture, infinite tiling, seed-consistent | Terrain Diffusion / InfiniteDiffusion; TerraFusion |
| Weave | Text/image -> tiling noise, normal maps, flowmaps | Latent diffusion with tiling conditioning |

---

## 3. Core v1 specification

A thin vertical slice that proves every seam end to end. Deliberately boring
functionality; the point is the interfaces.

### 3.1 Unit 1 — `elements-core::gpu`

Owns `wgpu::Device` and `Queue`. Provides:

- `FieldPool` — allocates and recycles 3D textures keyed by `(dims, format)`.
  Formats for v1: `R16Float`, `Rgba16Float`.
- `dispatch(shader, bindings, workgroups)` — compute dispatch wrapper.
- Device-lost and out-of-memory detection via `wgpu` error scopes.

Knows nothing about fire, water or terrain.

**Proves:** device acquisition on Metal/Vulkan/DX12; bounded memory growth
across frames.

### 3.2 Unit 2 — `elements-core::graph`

```rust
trait Node {
    fn sockets(&self) -> SocketSpec;
    fn eval(&self, ctx: &mut EvalCtx) -> Result<(), NodeError>;
}
```

- `Graph` is a DAG with topological evaluation and dirty-flag propagation.
- Serialized to a versioned `.elements` JSON document via serde.
- v1 node kinds: `ConstantField` (fill a 3D field with a scalar),
  `NoiseField` (curl noise, one compute shader, explicit seed), and one
  `Output` node.

**Proves:** socket type system, dirty propagation, and document round-trip. The
document format is the hardest thing to change later, so it is exercised from
day one.

### 3.3 Unit 3 — `elements-ipc`

**Control plane:** length-prefixed CBOR messages over a Unix domain socket
(macOS, Linux) or named pipe (Windows).

**Data plane:** POSIX shared memory / Windows file mapping holding a
double-buffered field, plus a sequence number the client polls. Two buffers; the
writer never blocks; the reader may skip frames, which is correct viewport
behavior.

Every control message carries `protocol_version`. On mismatch the daemon refuses
the connection and the addon shows "engine version mismatch, reinstall" rather
than deserializing garbage. This is v1 because version skew will be constant
during agent-driven development.

**Proves:** the riskiest seam in the project — shared memory across three
operating systems with a Python reader.

### 3.4 Unit 4 — `blender-elements` addon

- `Scene.elements` PropertyGroup.
- Operator to start and stop the daemon.
- Panel with a `.elements` graph file picker and engine status.
- Draw handler reading the shm field each viewport redraw and writing it into a
  Blender Volume object's grid.

Pure Python plus a small `elements_ipc` wheel for the shared-memory reader.

Packaging follows the project's Blender extension rules: `__init__.py` and
`blender_manifest.toml` at the ZIP root with nothing nested under a package
directory, relative imports throughout the addon package, wheels declared
per-platform in the manifest.

**Proves:** the install path and live viewport update.

### 3.5 Unit 5 — `elements-cli`

```
elements bake graph.elements --frames 1-100 --out ./vdb/
elements render-preview graph.elements --frame 1 --slice z=0.5 --out preview.png
```

No window, no Blender, no daemon; the same graph evaluation, writing `.vdb` via
`elements-io`. `render-preview` is a first-class deliverable, not a debugging
afterthought — it is what turns a broken shader into a diffable image.

**Proves:** the engine is genuinely headless, which is what lets agents test it
in CI.

### 3.6 Data flow, one frame

> **Not achieved in Core v1.** The diagram below is the intended design. What
> ships reads the frame for a status readout and a length check and then
> discards it; the geometry Blender renders comes from shelling out to the
> `elements` CLI, because bpy cannot build an OpenVDB grid in memory. The frame
> channel consequently has no production consumer and is exercised only by our
> own tests. Closing this needs an in-memory volume path, which arrives with
> Ember. Do not cite this section as delivered.

```
addon  --CBOR-->  daemon
                  graph.eval() -> FieldPool texture
                  readback -> shm buffer[n % 2], seq += 1
addon  <--shm---  draw handler reads seq, memcpy
                  -> Volume object grid -> viewport
```

### 3.7 Error handling

Three classes, each with defined behavior:

- **Protocol errors** (version mismatch, malformed CBOR): daemon logs, drops the
  connection; addon shows a banner.
- **GPU errors** (device lost, OOM): captured via `wgpu` error scopes, returned
  as a typed `EngineError` on the control plane; the daemon stays alive and
  retains the last good frame.
- **Graph errors** (cycle, socket type mismatch): surfaced as node-level errors
  in the panel; the valid subgraph still evaluates.

The addon must never raise an unhandled Python exception into Blender's draw
loop. The draw handler wraps everything and degrades to showing no volume.

### 3.8 Non-goals for Core v1

Undo integration; subgraphs; cross-frame GPU caching; Alembic; flipbook atlases;
ONNX inference; any real solver; a standalone GUI.

---

## 4. Testing strategy

| Tier | Runs | Covers | GPU |
|---|---|---|---|
| Unit | every commit | graph topology, dirty propagation, serde round-trip, CBOR codec, shm ring under concurrent access | no |
| Golden-field | every commit | `elements bake` on fixed seeds, compared to stored `.npy` within tolerance | lavapipe |
| Contract | every commit | daemon + Rust test client on the real protocol, including version mismatch, injected device-lost, malformed frames | no |
| Blender integration | pre-release | `blender --background --python` installs the built ZIP, starts the daemon, bakes one frame, asserts a non-empty Volume grid | no |

**CI GPU:** `wgpu` runs on **lavapipe** (software Vulkan). Slow but deterministic
and adequate for small golden fields, so agents get real signal without a GPU
runner. Physical-GPU runs happen locally before releases.

**Determinism is a design constraint.** Compute-shader floating-point results
vary across backends, so golden comparisons use tolerances and every stochastic
node takes an explicit seed. Anything that cannot be made reproducible gets a
property test instead (density conservation within epsilon, absence of NaNs,
values within bounds) — the approach the real solvers will rely on heavily.

---

## 5. Working method for coding agents

- **TDD is the delivery mechanism.** Every task is failing test -> implementation
  -> passing test. Agents cannot see rendered output; a task without an
  executable assertion will be reported complete while broken.
- **The five units are the parallelism boundary.** Once crate skeletons and trait
  signatures land, four of five proceed in parallel worktrees.
  **`elements-ipc` lands first** — the addon and daemon both depend on its wire
  format.
- **One agent, one crate, one worktree.** Cross-crate changes go through a spec
  amendment rather than an agent editing two crates at once.
- **The `.elements` document format is frozen by review.** Any PR changing it
  must bump the version and add a migration test.
- **Visual diffs on failure.** Golden tests store `render-preview` PNGs beside
  the `.npy` fields so a broken shader fails as an image diff.

The realistic failure mode of agent-driven GPU work is confident completion of
code that never ran on a GPU. The lavapipe tier and PNG diffs exist to make that
impossible to report as done.

---

## 6. Research directions

Genuine open problems this suite is positioned to address, ordered by
value-to-risk. None are Core v1 work.

1. **Neural super-resolution as an export-time operation.** Published upres work
   targets offline CFD; no artist tool ships "simulate at 128^3 interactively,
   bake at effective 1024^3." The unsolved part is temporal coherence —
   frame-independent upres flickers, and conditioning on the previous upres'd
   frame has not been demonstrated for art-directed pyro. Flagship feature.
2. **Video -> editable simulation.** WildSmoke, SmokeSVD and Vid2Fluid produce 3D
   smoke assets from single videos, but not a *re-simulatable state*. Recovering
   plausible initial conditions, forces and emitters so the artist receives a
   graph rather than a baked asset is open, and is the most differentiated thing
   the suite could do.
3. **GNS surrogate with classical fallback behind one interface.** Surrogates
   drift on unseen boundary conditions. Running the surrogate interactively and
   the solver on bake from a single graph, with a divergence metric shown to the
   artist, is both useful and a publishable evaluation.
4. **Learned sub-grid closures trained against our own solver.** Requires
   differentiable solver infrastructure (cf. 2025 adjoint-on-flow-maps work).
   Highest cost; the path to quality rather than speed.
5. **Flow-map advection (NFM / Cirrus lineage) on `wgpu`.** Not ML, but the
   largest available quality win for gas, and no open `wgpu` implementation
   exists.

### 6.1 Design position on ML

Every credible result in this domain — neural flow maps, GNS, super-resolution,
terrain diffusion — is ML wrapped around a classical formulation, and
pure-generative physics is not art-directable. This suite therefore uses ML
aggressively but always with a deterministic solver underneath. Recorded because
it is a deliberate departure from a "ML instead of traditional algorithms"
framing.

---

## 7. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Shared memory across macOS/Windows/Linux with a Python reader | High — could sink Core v1 | Built first as unit 3, with contract tests |
| `wgpu` compute performance vs CUDA for real solvers | High | Accept 2-3x slower; re-evaluate at Ember, not now |
| Model weights distribution and licensing | Medium | Separate downloadable packs; audit every pretrained model — much research code is non-commercial and cannot enter an Apache-2.0 engine |
| Agent-reported completion of code that never ran | Medium | lavapipe CI tier plus PNG golden diffs |
| Product naming trademark collisions | Low | "Ember" overlaps Ember.js (different field); trademark search required before any public release |
| Four products is years of work | Certain | Strict product-by-product specs; Core v1 is independently useful |

---

## 8. Roadmap

| Phase | Deliverable | Rough size |
|---|---|---|
| Core v1 | Thin vertical slice, all five units | 4-6 weeks |
| Ember v1 | Grid gas solver, VDB export, no ML | 8-12 weeks |
| Ember ML | Export-time upres; video -> sim | research-paced |
| Tide v1 | FLIP liquids, classical | follows Ember v1 |
| Strata v1 | Terrain, diffusion-heavy; parallelizable | parallel with Tide |
| Weave v1 | Procedural node assets | parallel with Tide |
| Tide ML | GNS surrogate preview | research-paced |

---

## 9. References

- Fluid Simulation on Neural Flow Maps — <https://arxiv.org/abs/2312.14635>
- Neural Monte Carlo Fluid Simulation, SIGGRAPH 2024 — <https://dl.acm.org/doi/10.1145/3641519.3657438>
- An Adjoint Method for Differentiable Fluid Simulation on Flow Maps — <https://arxiv.org/pdf/2511.01259>
- GNS: generalizable Graph Neural Network-based simulator — <https://arxiv.org/abs/2211.10228>
- Multiscale super-resolution reconstruction of fluid flows with deep neural networks — <https://pubs.aip.org/aip/adv/article/15/12/125209/3374082/Multiscale-super-resolution-reconstruction-of>
- InfiniteDiffusion / Terrain Diffusion — <https://arxiv.org/html/2512.08309>
- TerraFusion: joint terrain geometry and texture via latent diffusion — <https://arxiv.org/html/2505.04050v3>
- WildSmoke: dynamic 3D smoke assets from a single video — <https://arxiv.org/pdf/2509.11114>
- SmokeSVD: smoke reconstruction from a single view — <https://arxiv.org/pdf/2507.12156>
- Blender Python Wheels (extensions manual) — <https://docs.blender.org/manual/en/latest/advanced/extensions/python_wheels.html>
- `wgpu` — <https://wgpu.rs/>
- JangaFX (prior art; product names are their trademarks) — <https://jangafx.com/>
