// Conjugate gradients preconditioned by one multigrid V-cycle (2b-3c spec
// §3.5). Every scalar stays on the GPU: dot products are a `multiply` pass
// then a `Sum` reduction into a slot of `slots`, and the update kernels read
// the slots and compute α and β themselves, the same arithmetic in every
// thread.
//
// The operator is A x = h·L(x), which is negative definite, and the V-cycle
// approximates A⁻¹, so it is negative definite too. CG's formulas hold for
// any sign as long as M·A is positive definite, so they are used unchanged;
// only the guards test magnitudes. `q` holds residual(d, 0) = −A d, which
// saves a negation pass, so α = <r,z>/<d,Ad> = −rz/dq and r −= α·A d is
// r += α·q.
//
// The entry points share one binding layout; each binds only what it uses.

struct Pcg {
    dims: vec3<u32>,
    rz: u32,     // slot of <r,z> for the current iteration
    dq: u32,     // slot of <d,q>
    rz_new: u32, // slot of <r,z> after this iteration's update
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var a: texture_3d<f32>;
@group(0) @binding(1) var b: texture_3d<f32>;
@group(0) @binding(2) var x: texture_storage_3d<r32float, read_write>;
@group(0) @binding(3) var y: texture_storage_3d<r32float, read_write>;
@group(0) @binding(4) var<storage, read> slots: array<f32>;
@group(0) @binding(5) var<uniform> pcg: Pcg;

// Below this magnitude a dot product counts as converged, and the update
// adds nothing rather than dividing by it.
const TINY: f32 = 1e-30;

fn cell(gid: vec3<u32>) -> vec3<i32> {
    return vec3<i32>(gid);
}

// x = a·b, for a dot product's reduction.
@compute @workgroup_size(4, 4, 4)
fn multiply(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= pcg.dims)) {
        return;
    }
    let c = cell(gid);
    let v = textureLoad(a, c, 0).x * textureLoad(b, c, 0).x;
    textureStore(x, c, vec4<f32>(v, 0.0, 0.0, 0.0));
}

// x = 0.
@compute @workgroup_size(4, 4, 4)
fn zero(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= pcg.dims)) {
        return;
    }
    textureStore(x, cell(gid), vec4<f32>(0.0));
}

// x = a.
@compute @workgroup_size(4, 4, 4)
fn copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= pcg.dims)) {
        return;
    }
    let c = cell(gid);
    textureStore(x, c, vec4<f32>(textureLoad(a, c, 0).x, 0.0, 0.0, 0.0));
}

// p (x) += α·d (a); r (y) += α·q (b), with α = −rz/dq.
@compute @workgroup_size(4, 4, 4)
fn axpy_p_r(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= pcg.dims)) {
        return;
    }
    let rz = slots[pcg.rz];
    let dq = slots[pcg.dq];
    if (abs(rz) <= TINY || abs(dq) <= TINY) {
        return;
    }
    let alpha = -rz / dq;
    let c = cell(gid);
    let p = textureLoad(x, c).x + alpha * textureLoad(a, c, 0).x;
    let r = textureLoad(y, c).x + alpha * textureLoad(b, c, 0).x;
    textureStore(x, c, vec4<f32>(p, 0.0, 0.0, 0.0));
    textureStore(y, c, vec4<f32>(r, 0.0, 0.0, 0.0));
}

// d (x) = z (a) + β·d, with β = rz_new/rz.
@compute @workgroup_size(4, 4, 4)
fn update_d(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= pcg.dims)) {
        return;
    }
    let rz = slots[pcg.rz];
    var beta = 0.0;
    if (abs(rz) > TINY) {
        beta = slots[pcg.rz_new] / rz;
    }
    let c = cell(gid);
    let d = textureLoad(a, c, 0).x + beta * textureLoad(x, c).x;
    textureStore(x, c, vec4<f32>(d, 0.0, 0.0, 0.0));
}
