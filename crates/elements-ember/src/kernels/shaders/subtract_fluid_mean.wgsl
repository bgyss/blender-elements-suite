// Subtract a field's mean over fluid cells, given its sum from a reduction
// recorded earlier in the same batch. Solid cells hold 0 and are left so, so
// the sum is already a fluid sum; the fluid count is the level's cells less
// its solid count, reduced once per hierarchy into `solids[level.slot]`
// (0 without solids). A closed domain's Neumann problem is solvable only for
// a right-hand side with zero fluid sum.

@group(0) @binding(0) var field: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var<storage, read> sum: array<f32>;
@group(0) @binding(2) var<storage, read> solids: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var solid: texture_3d<f32>;
@group(0) @binding(5) var<uniform> level: LevelParams;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let p = vec3<i32>(gid);
    if (cell_solid(p)) {
        return;
    }
    let fluid = f32(params.dims.x * params.dims.y * params.dims.z) - solids[level.slot];
    if (fluid <= 0.0) {
        return;
    }
    let value = textureLoad(field, p).x - sum[0] / fluid;
    textureStore(field, p, vec4<f32>(value, 0.0, 0.0, 0.0));
}
