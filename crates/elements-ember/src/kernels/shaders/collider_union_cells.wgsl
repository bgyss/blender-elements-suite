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
