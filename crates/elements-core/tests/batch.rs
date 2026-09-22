use elements_core::gpu::{ComputeBatch, FieldDims, FieldPool, GpuContext, PipelineCache};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    dims: [u32; 3],
    value: f32,
}

fn uniform(gpu: &GpuContext, dims: FieldDims, value: f32) -> wgpu::Buffer {
    gpu.device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test-params"),
            contents: bytemuck::bytes_of(&Params {
                dims: [dims.x, dims.y, dims.z],
                value,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        })
}

/// Every dispatch in one batch sees the writes of the dispatches recorded
/// before it. Here: fill B with 2, then twice do A += B * 0.5. In order that
/// gives A = 2. Any reordering or lost write gives something else.
#[test]
fn dispatches_in_one_batch_run_in_order_and_see_earlier_writes() {
    let gpu = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let dims = FieldDims::new(8, 6, 5);
    let a = pool.acquire_zeroed(&gpu, &mut cache, dims).unwrap();
    let b = pool.acquire_zeroed(&gpu, &mut cache, dims).unwrap();

    let constant = cache
        .get_or_create(
            &gpu,
            "constant",
            include_str!("../src/gpu/shaders/constant.wgsl"),
            "main",
        )
        .unwrap();
    let accumulate = cache
        .get_or_create(
            &gpu,
            "accumulate",
            include_str!("../src/gpu/shaders/accumulate.wgsl"),
            "main",
        )
        .unwrap();

    let fill_params = uniform(&gpu, dims, 2.0);
    let add_params = uniform(&gpu, dims, 0.5);
    let fill_b = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &constant.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(b.view()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: fill_params.as_entire_binding(),
            },
        ],
    });
    let add_b_to_a = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &accumulate.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(a.view()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(b.view()),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: add_params.as_entire_binding(),
            },
        ],
    });

    let mut batch = ComputeBatch::new();
    batch.dispatch(&constant, &fill_b, dims);
    batch.dispatch(&accumulate, &add_b_to_a, dims);
    batch.dispatch(&accumulate, &add_b_to_a, dims);
    assert_eq!(batch.len(), 3);
    batch.submit(&gpu).unwrap();

    let values = a.read_back(&gpu).unwrap();
    assert!(values.iter().all(|&v| v == 2.0), "got {:?}", &values[..4]);
}

#[test]
fn an_empty_batch_submits_nothing_and_succeeds() {
    let gpu = GpuContext::new_headless().expect("no GPU adapter available");
    let batch = ComputeBatch::new();
    assert!(batch.is_empty());
    batch.submit(&gpu).unwrap();
}
