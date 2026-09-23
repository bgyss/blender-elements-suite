// Shared by both reduction passes, concatenated in front of each.
// `op` 0 is max |x|, 1 is sum. Both folds start from 0, which is the
// identity for each (|x| is never negative).

struct ReduceParams {
    dims: vec3<u32>,
    op: u32,
    count: u32, // partial pass: workgroups covering the field; final pass: partials to fold
    slot: u32,  // final pass: the target slot to write
    _pad0: u32,
    _pad1: u32,
};

const THREADS: u32 = 64u;    // must match THREADS in reduce.rs
const PER_THREAD: u32 = 64u; // must match PER_THREAD in reduce.rs

var<workgroup> scratch: array<f32, 64>;

fn combine(a: f32, b: f32) -> f32 {
    if (params.op == 0u) {
        return max(a, b);
    }
    return a + b;
}

// Fold `scratch` pairwise in a fixed order; the result ends in scratch[0].
// The order never depends on timing, so the result is bit-deterministic.
fn tree(lid: u32) {
    for (var stride = THREADS / 2u; stride > 0u; stride = stride / 2u) {
        if (lid < stride) {
            scratch[lid] = combine(scratch[lid], scratch[lid + stride]);
        }
        workgroupBarrier();
    }
}
