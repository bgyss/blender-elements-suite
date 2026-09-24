//! MGPCG: conjugate gradients preconditioned by one multigrid V-cycle from
//! zero (2b-3c spec §3.5; McAdams et al. 2010, as Mantaflow does).
//!
//! It solves the same problem as `v_cycles`, A p = div with A x = h·L(x)
//! over fluid cells, so `r = div − A p` is the multigrid residual. The
//! V-cycle from zero is symmetric, so it is a valid preconditioner, and CG
//! converges where plain V-cycles do not: around thin colliders, a one-cell
//! wall vanishes on coarse levels and plain cycles diverge slowly.
//!
//! Every scalar stays on the GPU (`shaders/pcg.wgsl`): the dot products
//! reduce into slots of one `ReduceTarget`, and the update kernels compute α
//! and β from the slots. `<r,z>` alternates between two slots, so an
//! iteration's update reads the old and the new value without a copy.

use std::sync::Arc;

use elements_core::gpu::{
    ComputeBatch, Field, FieldDims, FieldFormat, FieldPool, GpuContext, GpuError, PipelineCache,
    ReduceOp, ReduceTarget, reduce,
};

use super::multigrid::{FineResidual, FluidMean, Hierarchy, VCycle};
use super::{Bind, expect_dims, uniform_buffer};

const PCG: &str = include_str!("shaders/pcg.wgsl");

/// Slots of the dot products in the `ReduceTarget`. `<r,z>` alternates
/// between `RZ_A` and `RZ_B` from one iteration to the next.
const RZ_A: u32 = 0;
const DQ: u32 = 1;
const RZ_B: u32 = 2;
/// A fourth slot, unused.
const SLOTS: u32 = 4;

/// Matches `Pcg` in `shaders/pcg.wgsl`, 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PcgParams {
    dims: [u32; 3],
    rz: u32,
    dq: u32,
    rz_new: u32,
    _pad: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<PcgParams>() == 32);

fn params(
    gpu: &GpuContext,
    dims: FieldDims,
    rz: u32,
    rz_new: u32,
) -> Result<wgpu::Buffer, GpuError> {
    let p = PcgParams {
        dims: [dims.x, dims.y, dims.z],
        rz,
        dq: DQ,
        rz_new,
        _pad: [0; 2],
    };
    uniform_buffer(gpu, "ember-pcg", bytemuck::bytes_of(&p))
}

/// A bind group whose entries sit at the given binding numbers: the
/// `pcg.wgsl` entry points share one layout but each binds only what it uses.
fn sparse_group(
    gpu: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    entries: &[(u32, Bind<'_>)],
) -> Result<wgpu::BindGroup, GpuError> {
    let entries: Vec<wgpu::BindGroupEntry<'_>> = entries
        .iter()
        .map(|(binding, entry)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: match entry {
                Bind::Tex(field) => wgpu::BindingResource::TextureView(field.view()),
                Bind::Buf(buffer) => buffer.as_entire_binding(),
                Bind::View(view) => wgpu::BindingResource::TextureView(view),
            },
        })
        .collect();
    gpu.scoped(|| {
        gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ember-pcg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    })
}

/// One `pcg.wgsl` dispatch with its bind group.
struct Pass {
    pipe: Arc<wgpu::ComputePipeline>,
    group: wgpu::BindGroup,
    cells: FieldDims,
}

impl Pass {
    fn record(&self, batch: &mut ComputeBatch) {
        batch.dispatch(&self.pipe, &self.group, self.cells);
    }
}

fn pipeline(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    key: &'static str,
    entry: &str,
) -> Result<Arc<wgpu::ComputePipeline>, GpuError> {
    cache.get_or_create(gpu, key, PCG, entry)
}

/// Zeroing one field, bound once.
pub(crate) struct Zero(Pass);

impl Zero {
    pub(crate) fn new(
        gpu: &GpuContext,
        cache: &mut PipelineCache,
        field: &Field,
    ) -> Result<Self, GpuError> {
        let pipe = pipeline(gpu, cache, "ember.pcg.zero", "zero")?;
        let uniform = params(gpu, field.dims(), RZ_A, RZ_B)?;
        let group = sparse_group(
            gpu,
            &pipe,
            &[(2, Bind::Tex(field)), (5, Bind::Buf(&uniform))],
        )?;
        Ok(Self(Pass {
            pipe,
            group,
            cells: field.dims(),
        }))
    }

    pub(crate) fn record(&self, batch: &mut ComputeBatch) {
        self.0.record(batch);
    }
}

/// `iterations` of conjugate gradients on A p = div, preconditioned by one
/// V-cycle from zero, warm-started from what `p` holds: the first residual
/// is `div − A p`, so CG solves for the correction to `p`.
///
/// Scalars stay on the GPU. Five fields come from `pool` (r, z, d, q and a
/// scratch for products) and go back to it at the end; the batch is flushed
/// first, so later work cannot overtake the solve. In a closed domain the
/// fluid mean of r is removed each time r is formed, z's by the V-cycle, and
/// p's at the end.
#[allow(clippy::too_many_arguments)] // every arg is load-bearing; see the doc above.
pub fn mgpcg(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    pool: &mut FieldPool,
    h: &Hierarchy,
    p: &Field,
    div: &Field,
    iterations: u32,
) -> Result<(), GpuError> {
    let cells = h.fine().cells();
    expect_dims("mgpcg p", p, cells)?;
    expect_dims("mgpcg div", div, cells)?;
    if iterations == 0 {
        return Ok(());
    }
    let mut fields = Vec::with_capacity(5);
    for _ in 0..5 {
        fields.push(pool.acquire(gpu, cells, FieldFormat::R32Float)?);
    }
    let result =
        record(gpu, cache, batch, h, p, div, iterations, &fields).and_then(|()| batch.flush(gpu));
    for field in fields {
        pool.release(field);
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn record(
    gpu: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    h: &Hierarchy,
    p: &Field,
    div: &Field,
    iterations: u32,
    fields: &[Field],
) -> Result<(), GpuError> {
    let [r, z, d, q, scratch] = fields else {
        unreachable!("five fields");
    };
    let cells = h.fine().cells();
    let closed = h.closed();
    let slots = ReduceTarget::new(gpu, SLOTS)?;
    // Iteration i uses parities[i % 2]: <r,z> in RZ_A, the new one into RZ_B,
    // then the other way round.
    let parities = [
        params(gpu, cells, RZ_A, RZ_B)?,
        params(gpu, cells, RZ_B, RZ_A)?,
    ];
    let rz_new_slot = [RZ_B, RZ_A];

    let initial = FineResidual::new(gpu, cache, h, p, div, r)?;
    // q = residual(d, 0) = −A d, with `scratch` zeroed as the right-hand side.
    let apply = FineResidual::new(gpu, cache, h, d, scratch, q)?;
    let precondition = VCycle::new(gpu, cache, h, z, r)?;
    let zero_z = Zero::new(gpu, cache, z)?;
    let zero_scratch = Zero::new(gpu, cache, scratch)?;
    let r_mean = closed
        .then(|| FluidMean::new(gpu, cache, h, 0, r))
        .transpose()?;
    let p_mean = closed
        .then(|| FluidMean::new(gpu, cache, h, 0, p))
        .transpose()?;

    let multiply_pipe = pipeline(gpu, cache, "ember.pcg.multiply", "multiply")?;
    let multiply = |gpu: &GpuContext, a: &Field, b: &Field| -> Result<Pass, GpuError> {
        let group = sparse_group(
            gpu,
            &multiply_pipe,
            &[
                (0, Bind::Tex(a)),
                (1, Bind::Tex(b)),
                (2, Bind::Tex(scratch)),
                (5, Bind::Buf(&parities[0])),
            ],
        )?;
        Ok(Pass {
            pipe: multiply_pipe.clone(),
            group,
            cells,
        })
    };
    let rz_product = multiply(gpu, r, z)?;
    let dq_product = multiply(gpu, d, q)?;

    let copy_pipe = pipeline(gpu, cache, "ember.pcg.copy", "copy")?;
    let d_from_z = Pass {
        group: sparse_group(
            gpu,
            &copy_pipe,
            &[
                (0, Bind::Tex(z)),
                (2, Bind::Tex(d)),
                (5, Bind::Buf(&parities[0])),
            ],
        )?,
        pipe: copy_pipe,
        cells,
    };

    let axpy_pipe = pipeline(gpu, cache, "ember.pcg.axpy_p_r", "axpy_p_r")?;
    let update_pipe = pipeline(gpu, cache, "ember.pcg.update_d", "update_d")?;
    let mut axpy = Vec::with_capacity(2);
    let mut update = Vec::with_capacity(2);
    for parity in &parities {
        axpy.push(Pass {
            group: sparse_group(
                gpu,
                &axpy_pipe,
                &[
                    (0, Bind::Tex(d)),
                    (1, Bind::Tex(q)),
                    (2, Bind::Tex(p)),
                    (3, Bind::Tex(r)),
                    (4, Bind::Buf(slots.buffer())),
                    (5, Bind::Buf(parity)),
                ],
            )?,
            pipe: axpy_pipe.clone(),
            cells,
        });
        update.push(Pass {
            group: sparse_group(
                gpu,
                &update_pipe,
                &[
                    (0, Bind::Tex(z)),
                    (2, Bind::Tex(d)),
                    (4, Bind::Buf(slots.buffer())),
                    (5, Bind::Buf(parity)),
                ],
            )?,
            pipe: update_pipe.clone(),
            cells,
        });
    }

    let remove_r_mean = |cache: &mut PipelineCache, batch: &mut ComputeBatch| match &r_mean {
        Some(mean) => mean.record(gpu, cache, batch, h, r),
        None => Ok(()),
    };
    let dot = |cache: &mut PipelineCache,
               batch: &mut ComputeBatch,
               product: &Pass,
               slot: u32|
     -> Result<(), GpuError> {
        product.record(batch);
        reduce(gpu, cache, batch, scratch, ReduceOp::Sum, &slots, slot)
    };

    // z = M r: one V-cycle from zero.
    let apply_m = |cache: &mut PipelineCache, batch: &mut ComputeBatch| {
        zero_z.record(batch);
        precondition.record(gpu, cache, batch, h, z, 1)
    };

    // r = div − A p; z = M r; d = z; <r,z>.
    initial.record(batch);
    remove_r_mean(cache, batch)?;
    apply_m(cache, batch)?;
    d_from_z.record(batch);
    dot(cache, batch, &rz_product, RZ_A)?;

    for i in 0..iterations {
        let parity = (i % 2) as usize;
        if i > 0 {
            // One iteration (one V-cycle) per submission, as `v_cycles` does.
            batch.flush(gpu)?;
        }
        zero_scratch.record(batch);
        apply.record(batch);
        dot(cache, batch, &dq_product, DQ)?;
        axpy[parity].record(batch);
        remove_r_mean(cache, batch)?;
        if i + 1 < iterations {
            apply_m(cache, batch)?;
            dot(cache, batch, &rz_product, rz_new_slot[parity])?;
            update[parity].record(batch);
        }
    }
    if let Some(mean) = &p_mean {
        mean.record(gpu, cache, batch, h, p)?;
    }
    Ok(())
}
