//! Many compute dispatches, one queue submission.

use super::{FieldDims, GpuContext, GpuError, WORKGROUP};

/// Records compute dispatches and submits them together.
///
/// `dispatch_over_field` submits once per call, which suits a node that runs
/// one kernel a frame. A solver runs hundreds (every pressure iteration is
/// two), and per-submit overhead would then dominate its step time.
///
/// Dispatches run in the order they were recorded, and each one sees every
/// earlier dispatch's writes: WebGPU makes each dispatch its own usage scope
/// and orders storage writes between them.
///
/// Nothing touches the GPU until `flush` or `submit`, so dropping a batch
/// unsubmitted is harmless. Fields a batch reads or writes must stay alive,
/// and must not be released to a pool for reuse, until `submit` returns.
#[derive(Default)]
pub struct ComputeBatch {
    dispatches: Vec<Dispatch>,
    submissions: u32,
}

struct Dispatch {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    workgroups: [u32; 3],
}

impl ComputeBatch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one invocation per voxel of `dims`.
    pub fn dispatch(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        dims: FieldDims,
    ) {
        self.dispatch_workgroups(
            pipeline,
            bind_group,
            [
                dims.x.div_ceil(WORKGROUP),
                dims.y.div_ceil(WORKGROUP),
                dims.z.div_ceil(WORKGROUP),
            ],
        );
    }

    /// Record a dispatch of exactly `workgroups`, for kernels that do not
    /// run one invocation per voxel, such as a reduction.
    pub fn dispatch_workgroups(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: [u32; 3],
    ) {
        self.dispatches.push(Dispatch {
            pipeline: pipeline.clone(),
            bind_group: bind_group.clone(),
            workgroups,
        });
    }

    /// Number of dispatches recorded so far.
    pub fn len(&self) -> usize {
        self.dispatches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dispatches.is_empty()
    }

    /// Submit everything recorded so far as one submission, then keep
    /// recording. Dispatches recorded after a flush still see every write
    /// before it: the queue runs submissions in order.
    ///
    /// Fields the flushed work uses may still be in use on the GPU, so they
    /// must not go back to a pool until the batch's last submission.
    pub fn flush(&mut self, ctx: &GpuContext) -> Result<(), GpuError> {
        if self.dispatches.is_empty() {
            return Ok(());
        }
        let dispatches = std::mem::take(&mut self.dispatches);
        let submitted = ctx.scoped(|| {
            let mut encoder =
                ctx.device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("elements-batch"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("elements-batch"),
                    timestamp_writes: None,
                });
                for d in &dispatches {
                    pass.set_pipeline(&d.pipeline);
                    pass.set_bind_group(0, &d.bind_group, &[]);
                    let [x, y, z] = d.workgroups;
                    pass.dispatch_workgroups(x, y, z);
                }
            }
            ctx.queue().submit(Some(encoder.finish()));
        });
        // The closure always reaches `submit`, so count it even when the scope
        // reports an error: that error may be a stray one from unrelated
        // earlier work, and the queue may still be running this batch.
        // `Substep::abandon` relies on `submitted_any` to know it must wait
        // before releasing fields the batch reads.
        self.submissions += 1;
        submitted
    }

    /// Whether any recorded work has reached the queue.
    pub fn submitted_any(&self) -> bool {
        self.submissions > 0
    }

    /// Submit everything still recorded.
    pub fn submit(mut self, ctx: &GpuContext) -> Result<(), GpuError> {
        self.flush(ctx)
    }
}
