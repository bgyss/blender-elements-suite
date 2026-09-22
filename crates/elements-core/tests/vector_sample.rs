//! The hand-written staggered sampling library, checked against linear velocity
//! fields. Trilinear interpolation reproduces a linear function exactly, so any
//! error beyond float rounding is a bug in the sampling code.

use elements_core::gpu::{Axis, FieldDims, FieldPool, GpuContext, PipelineCache, StaggeredField};
use wgpu::util::DeviceExt;

const LIBRARY: &str = include_str!("../src/gpu/shaders/vector_sample.wgsl");

const PROBE: &str = r#"
@group(0) @binding(0) var vel_x: texture_3d<f32>;
@group(0) @binding(1) var vel_y: texture_3d<f32>;
@group(0) @binding(2) var vel_z: texture_3d<f32>;
@group(0) @binding(3) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> results: array<vec4<f32>>;

// points[i].w == 0: sample_velocity at xyz (cell units).
// points[i].w == 1: velocity_at_cell at the integer cell xyz.
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&points)) {
        return;
    }
    let p = points[i];
    if (p.w == 0.0) {
        results[i] = vec4<f32>(sample_velocity(vel_x, vel_y, vel_z, p.xyz), 0.0);
    } else {
        results[i] = vec4<f32>(velocity_at_cell(vel_x, vel_y, vel_z, vec3<i32>(p.xyz)), 0.0);
    }
}
"#;

fn u_at(p: [f32; 3]) -> f32 {
    1.0 + 2.0 * p[0] + 3.0 * p[1] + 5.0 * p[2]
}
fn v_at(p: [f32; 3]) -> f32 {
    7.0 - p[0] + 0.5 * p[1] + 2.0 * p[2]
}
fn w_at(p: [f32; 3]) -> f32 {
    -3.0 + 4.0 * p[0] - p[1] + 0.25 * p[2]
}

/// Where texel (0, 0, 0) of each face sits, in cell units.
fn face_offset(axis: Axis) -> [f32; 3] {
    match axis {
        Axis::X => [0.0, 0.5, 0.5],
        Axis::Y => [0.5, 0.0, 0.5],
        Axis::Z => [0.5, 0.5, 0.0],
    }
}

fn fill_face(ctx: &GpuContext, field: &StaggeredField, axis: Axis, f: fn([f32; 3]) -> f32) {
    let face = field.face(axis);
    let d = face.dims();
    let o = face_offset(axis);
    let mut values = Vec::with_capacity(d.voxel_count());
    for z in 0..d.z {
        for y in 0..d.y {
            for x in 0..d.x {
                values.push(f([x as f32 + o[0], y as f32 + o[1], z as f32 + o[2]]));
            }
        }
    }
    face.write(ctx, &values).unwrap();
}

fn linear_velocity(ctx: &GpuContext, pool: &mut FieldPool, cells: FieldDims) -> StaggeredField {
    let field = pool.acquire_staggered_uninit(ctx, cells).unwrap();
    fill_face(ctx, &field, Axis::X, u_at);
    fill_face(ctx, &field, Axis::Y, v_at);
    fill_face(ctx, &field, Axis::Z, w_at);
    field
}

/// Run the probe kernel over `points` and return one vec4 per point.
fn probe(
    ctx: &GpuContext,
    cache: &mut PipelineCache,
    field: &StaggeredField,
    points: &[[f32; 4]],
) -> Vec<[f32; 4]> {
    let source = format!("{LIBRARY}\n{PROBE}");
    let pipeline = cache
        .get_or_create(ctx, "test-vector-sample-probe", &source, "probe")
        .unwrap();
    let device = ctx.device();
    let size = std::mem::size_of_val(points) as u64;

    let point_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("probe-points"),
        contents: bytemuck::cast_slice(points),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let results = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe-results"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe-staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("probe-bind-group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(field.face(Axis::X).view()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(field.face(Axis::Y).view()),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(field.face(Axis::Z).view()),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: point_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: results.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("probe"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("probe"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((points.len() as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&results, 0, &staging, 0, size);
    ctx.queue().submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let out: Vec<[f32; 4]> = {
        let mapped = slice.get_mapped_range().unwrap();
        bytemuck::cast_slice(&mapped).to_vec()
    };
    staging.unmap();
    out
}

fn assert_near(got: [f32; 4], want: [f32; 3], at: [f32; 4]) {
    for axis in 0..3 {
        assert!(
            (got[axis] - want[axis]).abs() < 1e-3,
            "at {at:?}: component {axis} is {}, expected {}",
            got[axis],
            want[axis]
        );
    }
}

#[test]
fn sample_velocity_reproduces_a_linear_field_exactly() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let field = linear_velocity(&ctx, &mut pool, FieldDims::new(4, 5, 6));

    // Every point lies inside all three faces' sample hulls, [0.5, n - 0.5] on
    // each axis, so edge clamping never engages.
    let mut points = Vec::new();
    for x in [0.5, 1.25, 2.0, 3.5] {
        for y in [0.5, 2.75, 4.5] {
            for z in [0.5, 3.1, 5.5] {
                points.push([x, y, z, 0.0]);
            }
        }
    }

    let got = probe(&ctx, &mut cache, &field, &points);
    for (point, value) in points.iter().zip(got) {
        let p = [point[0], point[1], point[2]];
        assert_near(value, [u_at(p), v_at(p), w_at(p)], *point);
    }
}

#[test]
fn velocity_at_cell_is_the_velocity_at_the_cell_centre() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(4, 5, 6);
    let field = linear_velocity(&ctx, &mut pool, cells);

    let mut points = Vec::new();
    for x in 0..cells.x {
        for y in 0..cells.y {
            for z in 0..cells.z {
                points.push([x as f32, y as f32, z as f32, 1.0]);
            }
        }
    }

    let got = probe(&ctx, &mut cache, &field, &points);
    for (point, value) in points.iter().zip(got) {
        let centre = [point[0] + 0.5, point[1] + 0.5, point[2] + 0.5];
        assert_near(value, [u_at(centre), v_at(centre), w_at(centre)], *point);
    }
}
