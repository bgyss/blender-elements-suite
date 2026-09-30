use elements_ember::shack::{ShackParams, shack};

fn params() -> ShackParams {
    ShackParams {
        size: [1.2, 1.0, 1.0],
        plank_height: 0.1,
        thickness: 0.02,
        gap: 0.01,
        broken_fraction: 0.0,
        seed: 7,
    }
}

#[test]
fn an_unbroken_shack_has_every_plank_and_stays_inside_its_size() {
    let m = shack(&params());
    m.validate("test").unwrap();
    // 4 walls × floor(height / (plank + gap)) planks + 1 roof slab, 12 triangles each.
    let rows = (1.0f32 / 0.11).floor() as usize; // 9
    assert_eq!(m.triangle_count(), (4 * rows + 1) * 12);
    let (lo, hi) = m.bounds();
    // Walls span exactly ±size/2; the roof slab overhangs by the thickness.
    let eps = 1e-5;
    let t = 0.02;
    assert!(
        lo[0] >= -0.6 - t - eps && hi[0] <= 0.6 + t + eps,
        "x {lo:?} {hi:?}"
    );
    assert!(
        lo[1] >= -0.5 - t - eps && hi[1] <= 0.5 + t + eps,
        "y {lo:?} {hi:?}"
    );
    assert!(lo[2] >= -eps && hi[2] <= 1.0 + t + eps, "z {lo:?} {hi:?}");
}

#[test]
fn breakage_removes_planks_deterministically_per_seed() {
    let p = |seed, f| ShackParams {
        seed,
        broken_fraction: f,
        ..params()
    };
    let full = shack(&p(1, 0.0)).triangle_count();
    let a = shack(&p(1, 0.3));
    let b = shack(&p(1, 0.3));
    let c = shack(&p(2, 0.3));
    assert_eq!(a, b, "same seed, same shack");
    assert!(a.triangle_count() < full, "some planks are gone");
    assert_ne!(a, c, "a different seed breaks different planks");
    assert!(
        shack(&p(1, 1.0)).triangle_count() >= 12,
        "the roof always remains"
    );
}
