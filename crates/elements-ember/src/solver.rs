//! `ember.smoke_solver`: a dense-grid smoke solver (spec §2.4, §3).
//!
//! Per substep: emit, buoyancy, vorticity confinement, advect velocity,
//! project, and advect scalars with dissipation. Each frame first measures
//! the fastest face and picks its substep count by CFL.
//!
//! The projection's pressure solve is chosen by `pressure_solver`: red-black
//! Gauss–Seidel for `pressure_iterations` sweeps, or, on a multigrid
//! hierarchy built each substep, `pressure_cycles` V-cycles or MGPCG
//! iterations (2b-3c spec §3).

use elements_core::gpu::{
    Axis, ComputeBatch, Field, FieldDims, FieldFormat, FieldPool, GpuContext, GpuError,
    PipelineCache, StaggeredField,
};
use elements_core::graph::{DocError, EvalCtx, Node, NodeError, SocketSpec, SocketType, Value};
use serde::{Deserialize, Serialize};

use crate::boundaries::Boundaries;
use crate::cfl;
use crate::kernels::{self, Advection, Carried, Hierarchy, Pass, Solids, StepConstants, Uniforms};
use crate::node_util::{pair, take_listed};
use crate::params;

pub const KIND: &str = "ember.smoke_solver";

/// Names the solver's velocity state slot (a staggered field).
pub const VELOCITY: &str = "velocity";
/// Names the solver's density state slot (a cell-centred field).
pub const DENSITY: &str = "density";
const TEMPERATURE: &str = "temperature";
/// Holds p. The warm start stays valid when h changes (spec §4.3).
const PRESSURE: &str = "pressure";
const SLOTS: [&str; 4] = [VELOCITY, DENSITY, TEMPERATURE, PRESSURE];

const MAX_SUBSTEPS: u32 = 16;
const MAX_PRESSURE_ITERATIONS: u32 = 1000;
const MAX_PRESSURE_CYCLES: u32 = 64;
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
        // `pressure_iterations` is kept for documents that choose
        // Gauss–Seidel; the presets solve with MGPCG, decided 2026-09-24 in
        // `docs/bench/solver-gate.md` after the registered gate failed.
        let (pressure_iterations, max_substeps, pressure_cycles) = match self {
            // One substep: the largest cap whose 128³ frame fits in 100 ms
            // (spec §6), decided 2026-09-22 in `docs/bench/presets.md`.
            // MGPCG ×4: less than half Gauss–Seidel ×160's solve time at
            // 128³ and 256³, and more accurate in every scene and on the thin
            // plate, but short of the plate's 1e-3 target (`solver-gate.md`).
            Self::Preview => (160, 1, 4),
            // 480 iterations: ratio 0.0076 in `docs/bench/iteration-sweep.md`.
            // MGPCG ×10: passes every accuracy check of the solver gate,
            // the thin plate included (`solver-gate.md`).
            Self::Final => (480, 8, 10),
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
            wind_velocity: [0.0; 3],
            wind_rate: 0.0,
            pressure_solver: PressureSolver::Mgpcg,
            pressure_cycles,
        }
    }
}

/// Which method solves for pressure in the projection (2b-3c spec §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PressureSolver {
    /// Red-black Gauss–Seidel, `pressure_iterations` sweeps.
    #[default]
    GaussSeidel,
    /// `pressure_cycles` multigrid V-cycles.
    Multigrid,
    /// `pressure_cycles` iterations of conjugate gradients preconditioned by
    /// one V-cycle.
    Mgpcg,
}

/// One substep's pressure solve with its count: what `Substep::project` runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureSolve {
    /// Red-black Gauss–Seidel sweeps.
    GaussSeidel(u32),
    /// V-cycles on a hierarchy built for the substep.
    Multigrid(u32),
    /// MGPCG iterations on a hierarchy built for the substep.
    Mgpcg(u32),
}

/// `ember.smoke_solver`'s parameters, resolved: every field has a value.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct SolverParams {
    /// Cap on CFL substeps per frame (spec §3).
    pub max_substeps: u32,
    /// Target maximum cells travelled per substep.
    pub cfl: f32,
    /// Red-black Gauss–Seidel iterations per substep, when `pressure_solver`
    /// is `GaussSeidel`; 1..=1000.
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
    /// The ambient airflow, m/s: the velocity the air relaxes towards
    /// (2b-3c spec §6).
    pub wind_velocity: [f32; 3],
    /// How fast the air relaxes towards `wind_velocity`, 1/s, at least 0;
    /// 0 turns wind off.
    pub wind_rate: f32,
    /// The pressure solve: `gauss_seidel`, `multigrid` or `mgpcg`.
    pub pressure_solver: PressureSolver,
    /// V-cycles (`multigrid`) or PCG iterations (`mgpcg`) per substep;
    /// 1..=64. Gauss–Seidel ignores it.
    ///
    /// Preview's 4 under-converges around one-cell-thick colliders (a thin
    /// plate keeps 4.7e-3 of its divergence, against the gate's 1e-3
    /// target): scenes with thin walls should use `final` or set this to
    /// at least 10 (`docs/bench/solver-gate.md`).
    pub pressure_cycles: u32,
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
    /// 2b-3's wind, an acceleration. Accepted by the parser only so that
    /// validation can name its replacements instead of an unknown key.
    wind: Option<serde_json::Value>,
    wind_velocity: Option<[f32; 3]>,
    wind_rate: Option<f32>,
    pressure_solver: Option<PressureSolver>,
    pressure_cycles: Option<u32>,
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
    if doc.wind.is_some() {
        return Err(params::bad(
            KIND,
            "\"wind\" was replaced by \"wind_velocity\" (m/s) and \"wind_rate\" (1/s)",
        ));
    }
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
        wind_velocity: doc.wind_velocity.unwrap_or(preset.wind_velocity),
        wind_rate: doc.wind_rate.unwrap_or(preset.wind_rate),
        pressure_solver: doc.pressure_solver.unwrap_or(preset.pressure_solver),
        pressure_cycles: doc.pressure_cycles.unwrap_or(preset.pressure_cycles),
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
    if !(1..=MAX_PRESSURE_CYCLES).contains(&p.pressure_cycles) {
        return Err(params::bad(
            KIND,
            format!(
                "pressure_cycles must be 1..={MAX_PRESSURE_CYCLES}, got {}",
                p.pressure_cycles
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
    params::finite(KIND, "wind_velocity", &p.wind_velocity)?;
    params::finite(KIND, "wind_rate", &[p.wind_rate])?;
    if p.wind_rate < 0.0 {
        return Err(params::bad(
            KIND,
            format!("wind_rate must be at least 0, got {}", p.wind_rate),
        ));
    }
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
    /// The pressure solve `pressure_solver` names, with its count.
    pub fn pressure(&self) -> PressureSolve {
        match self.pressure_solver {
            PressureSolver::GaussSeidel => PressureSolve::GaussSeidel(self.pressure_iterations),
            PressureSolver::Multigrid => PressureSolve::Multigrid(self.pressure_cycles),
            PressureSolver::Mgpcg => PressureSolve::Mgpcg(self.pressure_cycles),
        }
    }

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
            wind_velocity: self.wind_velocity,
            wind_rate: self.wind_rate,
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

/// Velocity emission for one frame: the velocity weight (1/s) and the
/// target velocity (spec §3.3).
#[derive(Clone, Copy)]
pub struct Emission<'a> {
    pub weight: &'a Field,
    pub velocity: &'a StaggeredField,
}

/// A frame's inputs to the solver.
#[derive(Clone, Copy)]
pub struct Sources<'a> {
    /// Emission rates per second, at the domain's dims.
    pub density: &'a Field,
    pub temperature: &'a Field,
    /// Velocity emission, when an emitter's weight and target are connected.
    pub emission: Option<Emission<'a>>,
    /// The frame's collider, when its SDF and velocity are connected (spec §3.2).
    pub solids: Option<Solids<'a>>,
}

impl<'a> Sources<'a> {
    pub fn new(density: &'a Field, temperature: &'a Field) -> Self {
        Self {
            density,
            temperature,
            emission: None,
            solids: None,
        }
    }

    pub fn with_emission(self, emission: Emission<'a>) -> Self {
        Self {
            emission: Some(emission),
            ..self
        }
    }

    pub fn with_solids(self, solids: Solids<'a>) -> Self {
        Self {
            solids: Some(solids),
            ..self
        }
    }
}

/// One substep being recorded. Every stage records into one batch, so a
/// substep is one queue submission (spec §3).
///
/// A field a stage replaces cannot go back to the pool until the batch has
/// run, or a later stage could be handed the same texture while an earlier
/// dispatch still reads it. Replaced fields wait in `retired` until `submit`
/// or `abandon`.
pub struct Substep {
    constants: StepConstants,
    uniforms: Uniforms,
    batch: ComputeBatch,
    retired: Vec<Field>,
    advection: Advection,
    vorticity: bool,
    wind: bool,
}

impl Substep {
    pub fn new(gpu: &GpuContext, constants: &StepConstants) -> Result<Self, GpuError> {
        Ok(Self {
            constants: *constants,
            uniforms: Uniforms::new(gpu, constants)?,
            batch: ComputeBatch::new(),
            retired: Vec::new(),
            advection: constants.advection,
            vorticity: constants.vorticity > 0.0,
            wind: constants.wind_rate > 0.0,
        })
    }

    /// Everything before projection, four passes: emit, buoyancy, vorticity
    /// confinement, advect velocity. That is stages 1–3 of the piece 2
    /// spec's table, with 2b-1's confinement between forces and advection.
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
        if let Some(e) = sources.emission {
            kernels::blend_velocity(
                gpu,
                cache,
                &mut self.batch,
                u,
                &state.velocity,
                e.weight,
                e.velocity,
                sources.solids,
            )?;
        }
        kernels::buoyancy(
            gpu,
            cache,
            &mut self.batch,
            u,
            state.velocity.face(Axis::Z),
            &state.density,
            &state.temperature,
            sources.solids,
        )?;
        if self.wind {
            kernels::wind(
                gpu,
                cache,
                &mut self.batch,
                u,
                &state.velocity,
                sources.solids,
            )?;
        }
        if self.vorticity {
            self.confine_vorticity(gpu, cache, pool, state, sources.solids)?;
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
                sources.solids,
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
        solids: Option<Solids<'_>>,
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
        let recorded = kernels::curl(
            gpu,
            cache,
            &mut self.batch,
            u,
            &state.velocity,
            refs,
            solids,
        )
        .and_then(|()| {
            kernels::confine(
                gpu,
                cache,
                &mut self.batch,
                u,
                &state.velocity,
                refs,
                solids,
            )
        });
        // The batch may reference them whether or not recording finished.
        self.retired.extend(omega);
        recorded
    }

    /// Stage 4: make the velocity divergence-free, warm-starting from
    /// `state.pressure` with the solve `pressure` names. Faces touching
    /// `solids` end with the collider's velocity.
    ///
    /// Multigrid and MGPCG build a `Hierarchy` for this substep; its fields
    /// are retired with `div`, so they reach the pool only after the batch
    /// has run (`submit`) or been discarded (`abandon`). On an error the
    /// caller must `abandon` the substep: MGPCG returns its own scratch
    /// fields to the pool before returning an error, and only discarding
    /// the unflushed batch keeps those dispatches from ever running.
    pub fn project(
        &mut self,
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        pool: &mut FieldPool,
        state: &mut SolverState,
        pressure: PressureSolve,
        solids: Option<Solids<'_>>,
    ) -> Result<(), GpuError> {
        let div = pool.acquire(gpu, self.uniforms.cells(), FieldFormat::R32Float)?;
        let recorded = self.record_projection(gpu, cache, pool, state, &div, pressure, solids);
        // The batch may reference `div` whether or not recording finished.
        self.retired.push(div);
        recorded
    }

    /// `project`'s dispatches, into `div` and `state`.
    #[allow(clippy::too_many_arguments)]
    fn record_projection(
        &mut self,
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        pool: &mut FieldPool,
        state: &mut SolverState,
        div: &Field,
        pressure: PressureSolve,
        solids: Option<Solids<'_>>,
    ) -> Result<(), GpuError> {
        let u = &self.uniforms;
        kernels::divergence(gpu, cache, &mut self.batch, u, &state.velocity, div)?;
        let p = &state.pressure;
        match pressure {
            PressureSolve::GaussSeidel(iterations) => {
                kernels::solve_pressure(gpu, cache, &mut self.batch, u, p, div, iterations, solids)?
            }
            PressureSolve::Multigrid(count) | PressureSolve::Mgpcg(count) => {
                let h = Hierarchy::new(
                    gpu,
                    cache,
                    &mut self.batch,
                    pool,
                    &self.constants,
                    solids.map(|s| s.mask),
                )?;
                let solved = match pressure {
                    PressureSolve::Multigrid(_) => {
                        kernels::v_cycles(gpu, cache, &mut self.batch, &h, p, div, count)
                    }
                    _ => kernels::mgpcg(gpu, cache, &mut self.batch, pool, &h, p, div, count),
                };
                // The batch may reference the hierarchy whether or not
                // recording finished.
                self.retired.extend(h.into_fields());
                solved?;
            }
        }
        kernels::subtract_gradient(
            gpu,
            cache,
            &mut self.batch,
            &self.uniforms,
            &state.velocity,
            &state.pressure,
            solids,
        )
    }

    /// Stage 5: carry density and temperature through the projected velocity,
    /// never sampling the contents of `solids`.
    pub fn advect_scalars(
        &mut self,
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        pool: &mut FieldPool,
        state: &mut SolverState,
        solids: Option<Solids<'_>>,
    ) -> Result<(), GpuError> {
        let density = self.advect_grid(
            gpu,
            cache,
            pool,
            Carried::Density,
            &state.velocity,
            &state.density,
            solids,
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
            solids,
        )?;
        self.retired
            .push(std::mem::replace(&mut state.temperature, temperature));
        Ok(())
    }

    /// `src`, carried through `velocity` into a fresh pooled field. Scratch
    /// fields, and the new field if recording fails, are retired.
    #[allow(clippy::too_many_arguments)]
    fn advect_grid(
        &mut self,
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        pool: &mut FieldPool,
        carried: Carried,
        velocity: &StaggeredField,
        src: &Field,
        solids: Option<Solids<'_>>,
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
                solids,
            ),
            Advection::MacCormack => {
                self.maccormack(gpu, cache, pool, carried, velocity, src, &dst, solids)
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
        solids: Option<Solids<'_>>,
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
            solids,
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
                solids,
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
                solids,
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
    pressure: PressureSolve,
) -> Result<(), GpuError> {
    let mut step = Substep::new(gpu, constants)?;
    let recorded = step
        .pre_projection(gpu, cache, pool, state, sources)
        .and_then(|()| step.project(gpu, cache, pool, state, pressure, sources.solids))
        .and_then(|()| step.advect_scalars(gpu, cache, pool, state, sources.solids));
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
        let mut wanted = vec![0, 1];
        if pair(ctx, 2, 3)? {
            wanted.extend([2, 3]);
        }
        if pair(ctx, 4, 5)? {
            wanted.extend([4, 5]);
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

    fn step(
        &self,
        ctx: &mut EvalCtx<'_>,
        state: &mut SolverState,
        inputs: &[(u32, Value)],
    ) -> Result<(), NodeError> {
        let node = ctx.node_id();
        let find = |index: u32| inputs.iter().find(|(i, _)| *i == index).map(|(_, v)| v);
        let field = |index: u32| -> Result<&Field, NodeError> {
            find(index)
                .ok_or(NodeError::MissingInput { node, index })?
                .as_field()
                .map_err(|_| NodeError::TypeMismatch {
                    node,
                    index,
                    expected: SocketType::Field,
                })
        };
        let vector = |index: u32| -> Result<&StaggeredField, NodeError> {
            find(index)
                .ok_or(NodeError::MissingInput { node, index })?
                .as_vector_field()
                .map_err(|_| NodeError::TypeMismatch {
                    node,
                    index,
                    expected: SocketType::VectorField,
                })
        };
        let mut sources = Sources::new(field(0)?, field(1)?);
        if find(2).is_some() {
            sources = sources.with_emission(Emission {
                weight: field(2)?,
                velocity: vector(3)?,
            });
        }
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
    }

    /// The frame after the inputs are gathered: the CFL measurement, then
    /// every substep.
    fn step_frame(
        &self,
        ctx: &mut EvalCtx<'_>,
        state: &mut SolverState,
        sources: Sources<'_>,
    ) -> Result<(), NodeError> {
        let node = ctx.node_id();
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
        let constants = StepConstants {
            has_solids: sources.solids.is_some(),
            ..self
                .params
                .step_constants(ctx.dims(), (dt / f64::from(plan.count)) as f32, dx)
        };
        let pressure = self.params.pressure();
        ctx.with_gpu_pool(|gpu, cache, pool| {
            for _ in 0..plan.count {
                substep(gpu, cache, pool, state, sources, &constants, pressure)?;
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
            inputs: vec![
                SocketType::Field,
                SocketType::Field,
                SocketType::Field,
                SocketType::VectorField,
                SocketType::Field,
                SocketType::VectorField,
            ],
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
