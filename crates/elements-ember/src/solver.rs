//! `ember.smoke_solver`: a dense-grid smoke solver (spec §2.4, §3).
//!
//! Per substep: emit, buoyancy, vorticity confinement, advect velocity,
//! project, and advect scalars with dissipation. Each frame first measures
//! the fastest face and picks its substep count by CFL.

use elements_core::gpu::{
    Axis, ComputeBatch, Field, FieldDims, FieldFormat, FieldPool, GpuContext, GpuError,
    PipelineCache, StaggeredField,
};
use elements_core::graph::{DocError, EvalCtx, Node, NodeError, SocketSpec, SocketType, Value};
use serde::{Deserialize, Serialize};

use crate::boundaries::Boundaries;
use crate::cfl;
use crate::kernels::{self, Advection, Carried, Pass, StepConstants, Uniforms};
use crate::params;

pub const KIND: &str = "ember.smoke_solver";

const VELOCITY: &str = "velocity";
const DENSITY: &str = "density";
const TEMPERATURE: &str = "temperature";
/// Holds p. The warm start stays valid when h changes (spec §4.3).
const PRESSURE: &str = "pressure";
const SLOTS: [&str; 4] = [VELOCITY, DENSITY, TEMPERATURE, PRESSURE];

const MAX_SUBSTEPS: u32 = 16;
const MAX_PRESSURE_ITERATIONS: u32 = 1000;
const MAX_CFL: f32 = 10.0;

/// A quality preset (spec §6). It fills every field a document leaves unset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Quality {
    /// Interactive: 128³ within 100 ms a frame.
    #[default]
    Preview,
    /// Offline bakes: no time budget.
    Final,
}

impl Quality {
    /// The preset's full parameter set.
    pub fn params(self) -> SolverParams {
        let (pressure_iterations, max_substeps) = match self {
            // One substep: the largest cap whose 128³ frame fits in 100 ms
            // (spec §6), decided 2026-09-22 in `docs/bench/presets.md`.
            Self::Preview => (160, 1),
            // 480 iterations: ratio 0.0076 in `docs/bench/iteration-sweep.md`.
            Self::Final => (480, 8),
        };
        SolverParams {
            max_substeps,
            cfl: 1.0,
            pressure_iterations,
            advection: Advection::MacCormack,
            vorticity: 0.0,
            density_dissipation: 0.0,
            temperature_dissipation: 0.0,
            buoyancy_density: 0.0,
            // Provisional until 2b-3 maps parameters to Mantaflow's.
            buoyancy_temperature: 1.0,
            boundaries: Boundaries::default(),
        }
    }
}

/// `ember.smoke_solver`'s parameters, resolved: every field has a value.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct SolverParams {
    /// Cap on CFL substeps per frame (spec §3).
    pub max_substeps: u32,
    /// Target maximum cells travelled per substep.
    pub cfl: f32,
    /// Red-black Gauss–Seidel iterations per substep.
    pub pressure_iterations: u32,
    /// How velocity and scalars are advected (spec §4.1).
    pub advection: Advection,
    /// Vorticity confinement ε, 1/s; 0 turns it off (spec §4.4).
    pub vorticity: f32,
    /// Exponential decay of density, 1/s (spec §4.5).
    pub density_dissipation: f32,
    /// Exponential decay of temperature, 1/s.
    pub temperature_dissipation: f32,
    /// α: downward acceleration per unit density, m/s².
    pub buoyancy_density: f32,
    /// β: upward acceleration per unit temperature, m/s².
    pub buoyancy_temperature: f32,
    /// Which domain faces are open; the rest are walls (spec §4.2).
    pub boundaries: Boundaries,
}

impl Default for SolverParams {
    fn default() -> Self {
        Quality::default().params()
    }
}

/// A document's parameters as written. Every field is optional, and
/// `resolve_params` fills the gaps from the preset.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DocParams {
    quality: Option<Quality>,
    /// 2a's name for `max_substeps`, still accepted.
    substeps: Option<u32>,
    max_substeps: Option<u32>,
    cfl: Option<f32>,
    pressure_iterations: Option<u32>,
    advection: Option<Advection>,
    vorticity: Option<f32>,
    density_dissipation: Option<f32>,
    temperature_dissipation: Option<f32>,
    buoyancy_density: Option<f32>,
    buoyancy_temperature: Option<f32>,
    boundaries: Option<Boundaries>,
}

/// Parse `ember.smoke_solver`'s parameters from an untrusted document, fill
/// unset fields from the preset, and validate the result (spec §2).
pub fn resolve_params(params: &serde_json::Value) -> Result<SolverParams, DocError> {
    // `deny_unknown_fields` rejects misspelt keys; an absent `params` is `null`.
    let doc: DocParams = if params.is_null() {
        params::parse(KIND, &serde_json::json!({}))?
    } else {
        params::parse(KIND, params)?
    };
    if doc.substeps.is_some() && doc.max_substeps.is_some() {
        return Err(params::bad(
            KIND,
            "set max_substeps or its alias substeps, not both",
        ));
    }
    let preset = doc.quality.unwrap_or_default().params();
    let p = SolverParams {
        max_substeps: doc
            .max_substeps
            .or(doc.substeps)
            .unwrap_or(preset.max_substeps),
        cfl: doc.cfl.unwrap_or(preset.cfl),
        pressure_iterations: doc
            .pressure_iterations
            .unwrap_or(preset.pressure_iterations),
        advection: doc.advection.unwrap_or(preset.advection),
        vorticity: doc.vorticity.unwrap_or(preset.vorticity),
        density_dissipation: doc
            .density_dissipation
            .unwrap_or(preset.density_dissipation),
        temperature_dissipation: doc
            .temperature_dissipation
            .unwrap_or(preset.temperature_dissipation),
        buoyancy_density: doc.buoyancy_density.unwrap_or(preset.buoyancy_density),
        buoyancy_temperature: doc
            .buoyancy_temperature
            .unwrap_or(preset.buoyancy_temperature),
        boundaries: doc.boundaries.unwrap_or(preset.boundaries),
    };
    validate(&p)?;
    Ok(p)
}

fn validate(p: &SolverParams) -> Result<(), DocError> {
    if !(1..=MAX_SUBSTEPS).contains(&p.max_substeps) {
        return Err(params::bad(
            KIND,
            format!(
                "max_substeps must be 1..={MAX_SUBSTEPS}, got {}",
                p.max_substeps
            ),
        ));
    }
    if !(1..=MAX_PRESSURE_ITERATIONS).contains(&p.pressure_iterations) {
        return Err(params::bad(
            KIND,
            format!(
                "pressure_iterations must be 1..={MAX_PRESSURE_ITERATIONS}, got {}",
                p.pressure_iterations
            ),
        ));
    }
    params::finite(KIND, "cfl", &[p.cfl])?;
    if !(p.cfl > 0.0 && p.cfl <= MAX_CFL) {
        return Err(params::bad(
            KIND,
            format!("cfl must be in (0, {MAX_CFL}], got {}", p.cfl),
        ));
    }
    params::finite(
        KIND,
        "buoyancy",
        &[p.buoyancy_density, p.buoyancy_temperature],
    )?;
    let rates = [
        p.vorticity,
        p.density_dissipation,
        p.temperature_dissipation,
    ];
    params::finite(KIND, "vorticity and dissipation", &rates)?;
    if rates.iter().any(|&r| r < 0.0) {
        return Err(params::bad(
            KIND,
            "vorticity and dissipation rates must be at least 0",
        ));
    }
    Ok(())
}

impl SolverParams {
    /// Kernel constants for one substep of length `h`, in a domain of
    /// `cells` with voxel edge `dx`.
    pub fn step_constants(&self, cells: FieldDims, h: f32, dx: f32) -> StepConstants {
        StepConstants {
            alpha: self.buoyancy_density,
            beta: self.buoyancy_temperature,
            open_mask: self.boundaries.open_mask(),
            advection: self.advection,
            density_dissipation: self.density_dissipation,
            temperature_dissipation: self.temperature_dissipation,
            vorticity: self.vorticity,
            ..StepConstants::new(cells, h, dx)
        }
    }
}

/// Everything the solver carries from one step to the next.
pub struct SolverState {
    pub velocity: StaggeredField,
    pub density: Field,
    pub temperature: Field,
    /// p, kept as the next solve's warm start.
    pub pressure: Field,
}

impl SolverState {
    /// A still, empty domain.
    pub fn zeroed(
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        pool: &mut FieldPool,
        cells: FieldDims,
    ) -> Result<Self, GpuError> {
        let velocity = pool.acquire_staggered_zeroed(gpu, cache, cells)?;
        let mut fields = Vec::with_capacity(3);
        for _ in 0..3 {
            match pool.acquire_zeroed(gpu, cache, cells) {
                Ok(field) => fields.push(field),
                Err(e) => {
                    pool.release_staggered(velocity);
                    for field in fields {
                        pool.release(field);
                    }
                    return Err(e);
                }
            }
        }
        let [density, temperature, pressure]: [Field; 3] = match fields.try_into() {
            Ok(array) => array,
            Err(_) => unreachable!("exactly three fields were acquired"),
        };
        Ok(Self {
            velocity,
            density,
            temperature,
            pressure,
        })
    }

    pub fn release_to(self, pool: &mut FieldPool) {
        pool.release_staggered(self.velocity);
        pool.release(self.density);
        pool.release(self.temperature);
        pool.release(self.pressure);
    }

    /// The X, Y and Z faces, x-fastest, for tests and the speed gate.
    pub fn read_velocity(&self, gpu: &GpuContext) -> Result<[Vec<f32>; 3], GpuError> {
        Ok([
            self.velocity.face(Axis::X).read_back(gpu)?,
            self.velocity.face(Axis::Y).read_back(gpu)?,
            self.velocity.face(Axis::Z).read_back(gpu)?,
        ])
    }
}

/// Emission rates per second, at the domain's dims.
#[derive(Clone, Copy)]
pub struct Sources<'a> {
    pub density: &'a Field,
    pub temperature: &'a Field,
}

/// One substep being recorded. Every stage records into one batch, so a
/// substep is one queue submission (spec §3).
///
/// A field a stage replaces cannot go back to the pool until the batch has
/// run, or a later stage could be handed the same texture while an earlier
/// dispatch still reads it. Replaced fields wait in `retired` until `submit`
/// or `abandon`.
pub struct Substep {
    uniforms: Uniforms,
    batch: ComputeBatch,
    retired: Vec<Field>,
    advection: Advection,
    vorticity: bool,
}

impl Substep {
    pub fn new(gpu: &GpuContext, constants: &StepConstants) -> Result<Self, GpuError> {
        Ok(Self {
            uniforms: Uniforms::new(gpu, constants)?,
            batch: ComputeBatch::new(),
            retired: Vec::new(),
            advection: constants.advection,
            vorticity: constants.vorticity > 0.0,
        })
    }

    /// Stages 1–3: emit, buoyancy, vorticity confinement, advect velocity.
    pub fn pre_projection(
        &mut self,
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        pool: &mut FieldPool,
        state: &mut SolverState,
        sources: Sources<'_>,
    ) -> Result<(), GpuError> {
        let u = &self.uniforms;
        kernels::emit(
            gpu,
            cache,
            &mut self.batch,
            u,
            &state.density,
            sources.density,
        )?;
        kernels::emit(
            gpu,
            cache,
            &mut self.batch,
            u,
            &state.temperature,
            sources.temperature,
        )?;
        kernels::buoyancy(
            gpu,
            cache,
            &mut self.batch,
            u,
            state.velocity.face(Axis::Z),
            &state.density,
            &state.temperature,
        )?;
        if self.vorticity {
            self.confine_vorticity(gpu, cache, pool, state)?;
        }
        let cells = self.uniforms.cells();
        let mut faces = Vec::with_capacity(3);
        for axis in Axis::ALL {
            match self.advect_grid(
                gpu,
                cache,
                pool,
                Carried::Face(axis),
                &state.velocity,
                state.velocity.face(axis),
            ) {
                Ok(face) => faces.push(face),
                Err(e) => {
                    self.retired.extend(faces);
                    return Err(e);
                }
            }
        }
        let [x, y, z]: [Field; 3] = match faces.try_into() {
            Ok(array) => array,
            Err(_) => unreachable!("exactly three faces were advected"),
        };
        let advected = StaggeredField::from_faces(cells, [x, y, z])?;
        let old = std::mem::replace(&mut state.velocity, advected);
        self.retired.extend(old.into_faces());
        Ok(())
    }

    /// Vorticity confinement onto the velocity, through four pooled scratch
    /// fields for ω and |ω|.
    fn confine_vorticity(
        &mut self,
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        pool: &mut FieldPool,
        state: &SolverState,
    ) -> Result<(), GpuError> {
        let cells = self.uniforms.cells();
        let mut omega = Vec::with_capacity(4);
        for _ in 0..4 {
            match pool.acquire(gpu, cells, FieldFormat::R32Float) {
                Ok(field) => omega.push(field),
                Err(e) => {
                    self.retired.extend(omega);
                    return Err(e);
                }
            }
        }
        let refs = [&omega[0], &omega[1], &omega[2], &omega[3]];
        let u = &self.uniforms;
        let recorded = kernels::curl(gpu, cache, &mut self.batch, u, &state.velocity, refs)
            .and_then(|()| kernels::confine(gpu, cache, &mut self.batch, u, &state.velocity, refs));
        // The batch may reference them whether or not recording finished.
        self.retired.extend(omega);
        recorded
    }

    /// Stage 4: make the velocity divergence-free, warm-starting from `state.pressure`.
    pub fn project(
        &mut self,
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        pool: &mut FieldPool,
        state: &mut SolverState,
        iterations: u32,
    ) -> Result<(), GpuError> {
        let u = &self.uniforms;
        let div = pool.acquire(gpu, u.cells(), FieldFormat::R32Float)?;
        let recorded = kernels::divergence(gpu, cache, &mut self.batch, u, &state.velocity, &div)
            .and_then(|()| {
                kernels::solve_pressure(
                    gpu,
                    cache,
                    &mut self.batch,
                    u,
                    &state.pressure,
                    &div,
                    iterations,
                )
            })
            .and_then(|()| {
                kernels::subtract_gradient(
                    gpu,
                    cache,
                    &mut self.batch,
                    u,
                    &state.velocity,
                    &state.pressure,
                )
            });
        // The batch may reference `div` whether or not recording finished.
        self.retired.push(div);
        recorded
    }

    /// Stage 5: carry density and temperature through the projected velocity.
    pub fn advect_scalars(
        &mut self,
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        pool: &mut FieldPool,
        state: &mut SolverState,
    ) -> Result<(), GpuError> {
        let density = self.advect_grid(
            gpu,
            cache,
            pool,
            Carried::Density,
            &state.velocity,
            &state.density,
        )?;
        self.retired
            .push(std::mem::replace(&mut state.density, density));
        let temperature = self.advect_grid(
            gpu,
            cache,
            pool,
            Carried::Temperature,
            &state.velocity,
            &state.temperature,
        )?;
        self.retired
            .push(std::mem::replace(&mut state.temperature, temperature));
        Ok(())
    }

    /// `src`, carried through `velocity` into a fresh pooled field. Scratch
    /// fields, and the new field if recording fails, are retired.
    fn advect_grid(
        &mut self,
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        pool: &mut FieldPool,
        carried: Carried,
        velocity: &StaggeredField,
        src: &Field,
    ) -> Result<Field, GpuError> {
        let dst = pool.acquire(gpu, src.dims(), FieldFormat::R32Float)?;
        let recorded = match self.advection {
            Advection::SemiLagrangian => kernels::advect(
                gpu,
                cache,
                &mut self.batch,
                &self.uniforms,
                carried,
                Pass::SemiLagrangian,
                velocity,
                src,
                &dst,
            ),
            Advection::MacCormack => {
                self.maccormack(gpu, cache, pool, carried, velocity, src, &dst)
            }
        };
        match recorded {
            Ok(()) => Ok(dst),
            Err(e) => {
                self.retired.push(dst);
                Err(e)
            }
        }
    }

    /// Record MacCormack's three passes from `src` into `dst`.
    #[allow(clippy::too_many_arguments)]
    fn maccormack(
        &mut self,
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        pool: &mut FieldPool,
        carried: Carried,
        velocity: &StaggeredField,
        src: &Field,
        dst: &Field,
    ) -> Result<(), GpuError> {
        let fwd = pool.acquire(gpu, src.dims(), FieldFormat::R32Float)?;
        let bwd = match pool.acquire(gpu, src.dims(), FieldFormat::R32Float) {
            Ok(field) => field,
            Err(e) => {
                self.retired.push(fwd);
                return Err(e);
            }
        };
        let u = &self.uniforms;
        let recorded = kernels::advect(
            gpu,
            cache,
            &mut self.batch,
            u,
            carried,
            Pass::Forward,
            velocity,
            src,
            &fwd,
        )
        .and_then(|()| {
            kernels::advect(
                gpu,
                cache,
                &mut self.batch,
                u,
                carried,
                Pass::Backward,
                velocity,
                &fwd,
                &bwd,
            )
        })
        .and_then(|()| {
            kernels::maccormack(
                gpu,
                cache,
                &mut self.batch,
                u,
                carried,
                velocity,
                src,
                &fwd,
                &bwd,
                dst,
            )
        });
        // The batch may reference both whether or not recording finished.
        self.retired.push(fwd);
        self.retired.push(bwd);
        recorded
    }

    /// Run everything recorded, then return replaced fields to the pool.
    pub fn submit(self, gpu: &GpuContext, pool: &mut FieldPool) -> Result<(), GpuError> {
        let result = self.batch.submit(gpu);
        for field in self.retired {
            pool.release(field);
        }
        result
    }

    /// Discard everything still recorded without running it.
    ///
    /// Work already flushed may still be reading retired fields, so this
    /// waits for the GPU before releasing them (spec §5). The state passed to
    /// the stages may now hold fields that were never written, so the caller
    /// must discard the state too.
    pub fn abandon(self, gpu: &GpuContext, pool: &mut FieldPool) {
        if self.batch.submitted_any() {
            // The step has already failed; a second error adds nothing.
            let _ = gpu.wait();
        }
        for field in self.retired {
            pool.release(field);
        }
    }
}

/// One whole substep, stages 1–5, as one submission.
pub fn substep(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    pool: &mut FieldPool,
    state: &mut SolverState,
    sources: Sources<'_>,
    constants: &StepConstants,
    iterations: u32,
) -> Result<(), GpuError> {
    let mut step = Substep::new(gpu, constants)?;
    let recorded = step
        .pre_projection(gpu, cache, pool, state, sources)
        .and_then(|()| step.project(gpu, cache, pool, state, iterations))
        .and_then(|()| step.advect_scalars(gpu, cache, pool, state));
    match recorded {
        Ok(()) => step.submit(gpu, pool),
        Err(e) => {
            step.abandon(gpu, pool);
            Err(e)
        }
    }
}

#[derive(Debug, Clone)]
pub struct SmokeSolver {
    params: SolverParams,
}

impl SmokeSolver {
    /// Take the four state slots, or a zeroed state on the first step.
    fn take_state(ctx: &mut EvalCtx<'_>) -> Result<SolverState, NodeError> {
        let mut taken: Vec<(&'static str, Value)> = Vec::new();
        for slot in SLOTS {
            match ctx.take_state(slot) {
                Ok(Some(value)) => taken.push((slot, value)),
                Ok(None) => {}
                Err(e) => {
                    for (_, value) in taken {
                        ctx.release(value);
                    }
                    return Err(e);
                }
            }
        }
        if taken.is_empty() {
            let cells = ctx.dims();
            return ctx
                .with_gpu_pool(|gpu, cache, pool| SolverState::zeroed(gpu, cache, pool, cells));
        }

        let node = ctx.node_id();
        let missing = SLOTS
            .into_iter()
            .find(|slot| !taken.iter().any(|(name, _)| name == slot));
        // When every slot is present, name the first one whose value holds
        // the wrong `Value` variant, rather than always blaming `velocity`.
        let wrong_shape = taken.iter().find_map(|(slot, value)| {
            let matches_shape = if *slot == VELOCITY {
                matches!(value, Value::VectorField(_))
            } else {
                matches!(value, Value::Field(_))
            };
            if matches_shape { None } else { Some(*slot) }
        });
        let mut values = taken.into_iter().map(|(_, value)| value);
        match (values.next(), values.next(), values.next(), values.next()) {
            (
                Some(Value::VectorField(velocity)),
                Some(Value::Field(density)),
                Some(Value::Field(temperature)),
                Some(Value::Field(pressure)),
            ) if missing.is_none() => Ok(SolverState {
                velocity,
                density,
                temperature,
                pressure,
            }),
            (a, b, c, d) => {
                for value in [a, b, c, d].into_iter().flatten() {
                    ctx.release(value);
                }
                Err(NodeError::StateShape {
                    node,
                    slot: missing.or(wrong_shape).unwrap_or(VELOCITY),
                })
            }
        }
    }

    fn put_state(ctx: &mut EvalCtx<'_>, state: SolverState) -> Result<(), NodeError> {
        ctx.put_state(VELOCITY, Value::VectorField(state.velocity))?;
        ctx.put_state(DENSITY, Value::Field(state.density))?;
        ctx.put_state(TEMPERATURE, Value::Field(state.temperature))?;
        ctx.put_state(PRESSURE, Value::Field(state.pressure))
    }

    fn run(&self, ctx: &mut EvalCtx<'_>, state: &mut SolverState) -> Result<Vec<Value>, NodeError> {
        let density_source = ctx.take_input(0)?;
        let temperature_source = match ctx.take_input(1) {
            Ok(value) => value,
            Err(e) => {
                ctx.release(density_source);
                return Err(e);
            }
        };
        let stepped = self.step(ctx, state, &density_source, &temperature_source);
        ctx.release(density_source);
        ctx.release(temperature_source);
        stepped?;

        // The outputs are copies: the state stays in the store for the next
        // frame. Outputs nobody reads are not copied at all.
        let wanted: [bool; 3] = std::array::from_fn(|i| ctx.output_wanted(i as u32));
        ctx.with_gpu_pool(|gpu, _, pool| copy_outputs(gpu, pool, state, wanted))
    }

    fn step<'a>(
        &self,
        ctx: &mut EvalCtx<'_>,
        state: &mut SolverState,
        density_source: &'a Value,
        temperature_source: &'a Value,
    ) -> Result<(), NodeError> {
        let node = ctx.node_id();
        let field = |value: &'a Value, index: u32| -> Result<&'a Field, NodeError> {
            value.as_field().map_err(|_| NodeError::TypeMismatch {
                node,
                index,
                expected: SocketType::Field,
            })
        };
        let sources = Sources {
            density: field(density_source, 0)?,
            temperature: field(temperature_source, 1)?,
        };
        // Spec §3: one measurement per frame, from the entering state, so
        // the count is deterministic however the frame is reached.
        let dt = ctx.time().dt;
        let dx = ctx.voxel_size();
        let speed = ctx.with_gpu(|gpu, cache| cfl::measure_speed(gpu, cache, &state.velocity))?;
        let plan = cfl::plan_substeps(speed, dt, dx, self.params.cfl, self.params.max_substeps)
            .ok_or(NodeError::SolverDiverged { node })?;
        if plan.clamped {
            ctx.count_cfl_clamped();
        }
        let constants =
            self.params
                .step_constants(ctx.dims(), (dt / f64::from(plan.count)) as f32, dx);
        let iterations = self.params.pressure_iterations;
        ctx.with_gpu_pool(|gpu, cache, pool| {
            for _ in 0..plan.count {
                substep(gpu, cache, pool, state, sources, &constants, iterations)?;
            }
            Ok(())
        })
    }
}

impl Node for SmokeSolver {
    fn kind(&self) -> &'static str {
        KIND
    }

    fn sockets(&self) -> SocketSpec {
        SocketSpec {
            inputs: vec![SocketType::Field, SocketType::Field],
            outputs: vec![
                SocketType::Field,
                SocketType::Field,
                SocketType::VectorField,
            ],
        }
    }

    fn stateful(&self) -> bool {
        true
    }

    fn eval(&self, ctx: &mut EvalCtx<'_>) -> Result<Vec<Value>, NodeError> {
        let mut state = Self::take_state(ctx)?;
        match self.run(ctx, &mut state) {
            Ok(outputs) => {
                Self::put_state(ctx, state)?;
                Ok(outputs)
            }
            Err(e) => {
                // A half-run step may hold fields that were never written.
                // Never write it back; the timeline starts over from its cache.
                ctx.with_gpu_pool(|_, _, pool| {
                    state.release_to(pool);
                    Ok(())
                })?;
                Err(e)
            }
        }
    }
}

pub(crate) fn build(params: &serde_json::Value) -> Result<Box<dyn Node>, DocError> {
    Ok(Box::new(SmokeSolver {
        params: resolve_params(params)?,
    }))
}

/// Pooled copies of the wanted outputs, in socket order, with a
/// `Value::Scalar(0.0)` placeholder in each unwanted slot (see
/// `EvalCtx::output_wanted`). On failure, copies already made go back.
fn copy_outputs(
    gpu: &GpuContext,
    pool: &mut FieldPool,
    state: &SolverState,
    wanted: [bool; 3],
) -> Result<Vec<Value>, GpuError> {
    let mut outputs: Vec<Value> = Vec::with_capacity(3);
    for (index, wanted) in wanted.into_iter().enumerate() {
        let copied = if !wanted {
            Ok(Value::Scalar(0.0))
        } else {
            match index {
                0 => pool.duplicate(gpu, &state.density).map(Value::Field),
                1 => pool.duplicate(gpu, &state.temperature).map(Value::Field),
                _ => duplicate_velocity(gpu, pool, &state.velocity).map(Value::VectorField),
            }
        };
        match copied {
            Ok(value) => outputs.push(value),
            Err(e) => {
                for value in outputs {
                    value.release_to(pool);
                }
                return Err(e);
            }
        }
    }
    Ok(outputs)
}

/// A pooled copy of all three faces. On failure, faces already copied go back to the pool.
fn duplicate_velocity(
    gpu: &GpuContext,
    pool: &mut FieldPool,
    velocity: &StaggeredField,
) -> Result<StaggeredField, GpuError> {
    let mut faces = Vec::with_capacity(3);
    for axis in Axis::ALL {
        match pool.duplicate(gpu, velocity.face(axis)) {
            Ok(face) => faces.push(face),
            Err(e) => {
                for face in faces {
                    pool.release(face);
                }
                return Err(e);
            }
        }
    }
    let [x, y, z]: [Field; 3] = match faces.try_into() {
        Ok(array) => array,
        Err(_) => unreachable!("exactly three faces were copied"),
    };
    StaggeredField::from_faces(velocity.cells(), [x, y, z])
}
