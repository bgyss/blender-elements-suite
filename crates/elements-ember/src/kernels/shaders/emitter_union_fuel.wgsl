// ember.emitter_union, fuel: the rates add (2b-4 spec §4.2).

struct Grid {
    dims: vec3<u32>,
    axis: u32,
};

@group(0) @binding(0) var a: texture_3d<f32>;
@group(0) @binding(1) var b: texture_3d<f32>;
@group(0) @binding(2) var out: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var<uniform> grid: Grid;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= grid.dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    let f = textureLoad(a, p, 0).x + textureLoad(b, p, 0).x;
    textureStore(out, p, vec4<f32>(f, 0.0, 0.0, 0.0));
}
