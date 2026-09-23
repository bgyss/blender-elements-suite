//! Reducing a field to one number on the GPU: the max of |x|, or the sum.

use wgpu::util::DeviceExt;

use super::{ComputeBatch, Field, FieldFormat, GpuContext, GpuError, PipelineCache};

const PARTIAL: &str = concat!(
    include_str!("shaders/reduce_common.wgsl"),
    include_str!("shaders/reduce_partial.wgsl"),
);
const FINAL: &str = concat!(
    include_str!("shaders/reduce_common.wgsl"),
    include_str!("shaders/reduce_final.wgsl"),
);

/// Invocations per workgroup; `THREADS` in `reduce_common.wgsl`.
const THREADS: u32 = 64;
/// Voxels each invocation folds before the workgroup's tree; `PER_THREAD`.
const PER_THREAD: u32 = 64;
/// WebGPU's per-dimension limit on workgroup counts.
const MAX_GROUPS_PER_DIM: u32 = 65535;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// The largest |x|.
    MaxAbs,
    /// The sum of x.
    Sum,
}

/// Matches `ReduceParams` in `reduce_common.wgsl`, 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReduceParams {
    dims: [u32; 3],
    op: u32,
    count: u32,
    slot: u32,
    _pad: [u32; 2],
}

// `ReduceParams` in `shaders/reduce_common.wgsl` is 32 bytes; a field added
// here without its WGSL twin (or padding) fails the build instead of
// silently shifting every uniform the shader reads.
const _: () = assert!(std::mem::size_of::<ReduceParams>() == 32);

/// Where reductions land: `slots` values in one GPU buffer, zero at creation.
///
/// A later kernel in the same batch can bind `buffer()` as
/// `var<storage, read>` and use a result without a round trip to the CPU.
pub struct ReduceTarget {
    buffer: wgpu::Buffer,
    slots: u32,
}

impl ReduceTarget {
    pub fn new(ctx: &GpuContext, slots: u32) -> Result<Self, GpuError> {
        let buffer = ctx.scoped(|| {
            ctx.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("reduce-target"),
                size: u64::from(slots.max(1)) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        })?;
        Ok(Self { buffer, slots })
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn slots(&self) -> u32 {
        self.slots
    }

    /// Every slot, read back to the CPU. Call it after submitting the batch
    /// that wrote them; it blocks until the GPU has finished.
    pub fn read(&self, ctx: &GpuContext) -> Result<Vec<f32>, GpuError> {
        let size = self.buffer.size();
        let staging = ctx.scoped(|| {
            let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("reduce-readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder =
                ctx.device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("reduce-readback"),
                    });
            encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, size);
            ctx.queue().submit(Some(encoder.finish()));
            staging
        })?;

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        ctx.scoped(|| {
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            })
        })?;
        ctx.wait()?;
        rx.recv()
            .map_err(|e| GpuError::Validation(e.to_string()))?
            .map_err(|e| GpuError::Validation(e.to_string()))?;
        let values = {
            let mapped = slice
                .get_mapped_range()
                .map_err(|e| GpuError::Validation(e.to_string()))?;
            let floats: &[f32] = bytemuck::cast_slice(&mapped);
            floats[..self.slots as usize].to_vec()
        };
        ctx.scoped(|| staging.unmap())?;
        Ok(values)
    }
}

/// Record a reduction of `src` into `target`'s `slot`. Nothing runs until
/// `batch` is submitted.
///
/// Each workgroup folds a fixed run of voxels into a partial, and one
/// workgroup then folds the partials, all in a fixed order. So the result is
/// bit-identical from run to run on one backend. A sum's rounding differs
/// from a sequential CPU sum.
pub fn reduce(
    ctx: &GpuContext,
    cache: &mut PipelineCache,
    batch: &mut ComputeBatch,
    src: &Field,
    op: ReduceOp,
    target: &ReduceTarget,
    slot: u32,
) -> Result<(), GpuError> {
    if src.format() != FieldFormat::R32Float {
        return Err(GpuError::Validation(format!(
            "reduce: a {:?} field, expected R32Float",
            src.format()
        )));
    }
    if slot >= target.slots {
        return Err(GpuError::Validation(format!(
            "reduce: slot {slot} of a {}-slot target",
            target.slots
        )));
    }
    let dims = src.dims();
    let per_group = u64::from(THREADS * PER_THREAD);
    let groups = u32::try_from((dims.voxel_count() as u64).div_ceil(per_group))
        .map_err(|_| GpuError::Validation(format!("reduce: {dims:?} is too large")))?;
    let partial = cache.get_or_create(ctx, "core.reduce.partial", PARTIAL, "main")?;
    let fold = cache.get_or_create(ctx, "core.reduce.final", FINAL, "main")?;
    let params = ReduceParams {
        dims: [dims.x, dims.y, dims.z],
        op: match op {
            ReduceOp::MaxAbs => 0,
            ReduceOp::Sum => 1,
        },
        count: groups,
        slot,
        _pad: [0; 2],
    };
    // The bind groups hold the partials and uniform buffers alive until the
    // batch has run, so the handles can go out of scope here.
    let (partial_group, final_group) = ctx.scoped(|| {
        let partials = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("reduce-partials"),
            size: u64::from(groups) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let uniform = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("reduce-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let partial_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reduce-partial"),
            layout: &partial.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: partials.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let final_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reduce-final"),
            layout: &fold.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: partials.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: target.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        (partial_group, final_group)
    })?;
    batch.dispatch_workgroups(
        &partial,
        &partial_group,
        [
            groups.min(MAX_GROUPS_PER_DIM),
            groups.div_ceil(MAX_GROUPS_PER_DIM),
            1,
        ],
    );
    batch.dispatch_workgroups(&fold, &final_group, [1, 1, 1]);
    Ok(())
}
