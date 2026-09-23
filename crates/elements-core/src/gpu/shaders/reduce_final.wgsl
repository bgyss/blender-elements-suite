// Pass 2: one workgroup folds every partial into the target slot.

@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> slots: array<f32>;
@group(0) @binding(2) var<uniform> params: ReduceParams;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lid: u32) {
    var acc = 0.0;
    for (var i = lid; i < params.count; i = i + THREADS) {
        acc = combine(acc, partials[i]);
    }
    scratch[lid] = acc;
    workgroupBarrier();
    tree(lid);
    if (lid == 0u) {
        slots[params.slot] = scratch[0];
    }
}
