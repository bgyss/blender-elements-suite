// The velocity weight on face `p` of `axis`'s grid: the larger weight of the
// two cells it separates, with cells beyond the domain clamped in
// (2b-2 spec §2.4, §3.3).
fn face_weight(w: texture_3d<f32>, axis: u32, p: vec3<i32>, dims: vec3<u32>) -> f32 {
    var e = vec3<i32>(0);
    e[axis] = 1;
    let last = vec3<i32>(dims) - vec3<i32>(1);
    let below = textureLoad(w, clamp(p - e, vec3<i32>(0), last), 0).x;
    let above = textureLoad(w, clamp(p, vec3<i32>(0), last), 0).x;
    return max(below, above);
}
