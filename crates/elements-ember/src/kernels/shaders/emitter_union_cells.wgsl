// ember.emitter_union, cell pass: rates add, the weight is the maximum.

struct Grid {
    dims: vec3<u32>,
    axis: u32,
};

@group(0) @binding(0) var d1: texture_3d<f32>;
@group(0) @binding(1) var t1: texture_3d<f32>;
@group(0) @binding(2) var w1: texture_3d<f32>;
@group(0) @binding(3) var d2: texture_3d<f32>;
@group(0) @binding(4) var t2: texture_3d<f32>;
@group(0) @binding(5) var w2: texture_3d<f32>;
@group(0) @binding(6) var density: texture_storage_3d<r32float, write>;
@group(0) @binding(7) var temperature: texture_storage_3d<r32float, write>;
@group(0) @binding(8) var weight: texture_storage_3d<r32float, write>;
@group(0) @binding(9) var<uniform> grid: Grid;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= grid.dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    let d = textureLoad(d1, p, 0).x + textureLoad(d2, p, 0).x;
    let t = textureLoad(t1, p, 0).x + textureLoad(t2, p, 0).x;
    let w = max(textureLoad(w1, p, 0).x, textureLoad(w2, p, 0).x);
    textureStore(density, p, vec4<f32>(d, 0.0, 0.0, 0.0));
    textureStore(temperature, p, vec4<f32>(t, 0.0, 0.0, 0.0));
    textureStore(weight, p, vec4<f32>(w, 0.0, 0.0, 0.0));
}
