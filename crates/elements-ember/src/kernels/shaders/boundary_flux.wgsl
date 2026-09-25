// What one substep removes from a scalar before its advection (2b-3c spec
// §5), per cell: for each open domain face beside the cell, the first-order
// upwind outflow q · max(u_n, 0) · h / dx, with u_n the face's velocity
// positive outward; plus q itself in a solid cell, which the advection
// zeroes. Inflow brings ambient 0 and counts nothing; wall faces carry
// nothing. The per-cell values are summed by a reduction afterwards.

@group(0) @binding(0) var vel_x: texture_3d<f32>;
@group(0) @binding(1) var vel_y: texture_3d<f32>;
@group(0) @binding(2) var vel_z: texture_3d<f32>;
@group(0) @binding(3) var field: texture_3d<f32>;
@group(0) @binding(4) var removed: texture_storage_3d<r32float, write>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var solid: texture_3d<f32>;

// The normal velocity at face `f` of `axis`'s face grid.
fn normal_velocity(axis: u32, f: vec3<i32>) -> f32 {
    if (axis == 0u) {
        return textureLoad(vel_x, f, 0).x;
    }
    if (axis == 1u) {
        return textureLoad(vel_y, f, 0).x;
    }
    return textureLoad(vel_z, f, 0).x;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid >= params.dims)) {
        return;
    }
    let c = vec3<i32>(gid);
    let q = textureLoad(field, c, 0).x;
    if (cell_solid(c)) {
        textureStore(removed, c, vec4<f32>(q, 0.0, 0.0, 0.0));
        return;
    }
    var speed = 0.0;
    for (var a = 0u; a < 3u; a = a + 1u) {
        if (gid[a] == 0u && is_open(a, 0u)) {
            // The low face is face 0 of this cell; outward is −axis.
            speed += max(-normal_velocity(a, c), 0.0);
        }
        if (gid[a] == params.dims[a] - 1u && is_open(a, 1u)) {
            var e = vec3<i32>(0);
            e[a] = 1;
            speed += max(normal_velocity(a, c + e), 0.0);
        }
    }
    textureStore(removed, c, vec4<f32>(q * speed * params.h * params.inv_dx, 0.0, 0.0, 0.0));
}
