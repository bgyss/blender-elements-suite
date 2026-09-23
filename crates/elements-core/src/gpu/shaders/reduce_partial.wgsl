// Pass 1: each workgroup folds THREADS * PER_THREAD voxels into one partial.
// Workgroups are laid out in 2D, because one dimension allows at most 65535.

@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<f32>;
@group(0) @binding(2) var<uniform> params: ReduceParams;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let group = wid.x + wid.y * groups.x;
    let total = params.dims.x * params.dims.y * params.dims.z;
    let base = group * THREADS * PER_THREAD;
    var acc = 0.0;
    for (var n = 0u; n < PER_THREAD; n = n + 1u) {
        let i = base + n * THREADS + lid;
        if (i < total) {
            let x = i % params.dims.x;
            let y = (i / params.dims.x) % params.dims.y;
            let z = i / (params.dims.x * params.dims.y);
            var v = textureLoad(src, vec3<i32>(vec3<u32>(x, y, z)), 0).x;
            if (params.op == 0u) {
                v = abs(v);
            }
            acc = combine(acc, v);
        }
    }
    scratch[lid] = acc;
    workgroupBarrier();
    tree(lid);
    if (lid == 0u && group < params.count) {
        partials[group] = scratch[0];
    }
}
