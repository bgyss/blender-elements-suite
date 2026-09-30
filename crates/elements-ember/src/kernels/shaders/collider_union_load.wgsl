// ember.collider_union, load pass: the larger wood load (FT4 spec §3.1).

struct Grid {
    dims: vec3<u32>,
    axis: u32,
};

@group(0) @binding(0) var l1: texture_3d<f32>;
@group(0) @binding(1) var l2: texture_3d<f32>;
@group(0) @binding(2) var out_load: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var<uniform> grid: Grid;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= grid.dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    textureStore(out_load, p, vec4<f32>(max(textureLoad(l1, p, 0).x, textureLoad(l2, p, 0).x), 0.0, 0.0, 0.0));
}
