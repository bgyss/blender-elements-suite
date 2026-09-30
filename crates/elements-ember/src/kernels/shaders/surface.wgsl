// Surface ignition helpers (FT4 spec §3.3). The kernel declares `params`
// and `solid`. A "fluid" cell is inside the domain and not solid.

fn neighbour(p: vec3<i32>, n: u32) -> vec3<i32> {
    var d = vec3<i32>(0);
    d[n / 2u] = select(-1, 1, (n & 1u) == 1u);
    return p + d;
}

fn in_domain(c: vec3<i32>) -> bool {
    return all(c >= vec3<i32>(0)) && all(c < vec3<i32>(params.dims));
}

fn is_fluid(c: vec3<i32>) -> bool {
    return in_domain(c) && !cell_solid(c);
}

fn fluid_count(p: vec3<i32>) -> u32 {
    var count = 0u;
    for (var n = 0u; n < 6u; n = n + 1u) {
        if (is_fluid(neighbour(p, n))) {
            count = count + 1u;
        }
    }
    return count;
}
