// The mass correction's scaling pass (2b-3c spec §5): q ×= s with
// s = decay · (M₀ − OUT) / M₁, clamped to [MIN_SCALE, MAX_SCALE]. Every
// thread reads the three slots and computes s itself, so no one-thread
// kernel is needed. `params` is the carried scalar's uniform, whose `decay`
// the advection has already applied. A field that held a negative cell
// before advection (slot NEG above 0) is left alone: the rescale assumes
// q ≥ 0, and with both signs its sum is no measure of the error.

@group(0) @binding(0) var field: texture_storage_3d<r32float, read_write>;
@group(0) @binding(1) var<storage, read> slots: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// Slots of the correction's `ReduceTarget`; `conserve.rs` mirrors them.
const M0: u32 = 0u;
const OUT: u32 = 1u;
const M1: u32 = 2u;
const NEG: u32 = 3u;
// A safeguard, not a working range: `MIN_SCALE` and `MAX_SCALE`.
const MIN_SCALE: f32 = 0.9;
const MAX_SCALE: f32 = 1.1;
// Below this the field is empty and nothing is scaled.
const EMPTY: f32 = 1e-12;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let kept = slots[M0] - slots[OUT];
    let after = slots[M1];
    if (after <= EMPTY || kept <= 0.0 || slots[NEG] > 0.0) {
        return;
    }
    let s = clamp(params.decay * kept / after, MIN_SCALE, MAX_SCALE);
    let c = vec3<i32>(gid);
    textureStore(field, c, vec4<f32>(textureLoad(field, c).x * s, 0.0, 0.0, 0.0));
}
