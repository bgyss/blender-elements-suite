mod common;

use common::*;
use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, PipelineCache};
use elements_ember::collider::{ColliderFields, ColliderParams, fill_collider};
use elements_ember::transform::{Key, Rotate, Shape, Transform};

const SPF: f64 = 1.0 / 24.0;

/// SDF of `shape` at every cell centre of a 32³, 2 m domain.
fn sdf(shape: Shape, rotate: Option<Rotate>, at: [f32; 3]) -> Vec<f32> {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(32, 32, 32);
    let out = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    let v = pool.acquire_staggered_uninit(&gpu, cells).unwrap();
    let params = ColliderParams {
        shape,
        transform: Transform {
            keys: vec![Key {
                frame: 0.0,
                translate: at,
                rotate,
            }],
        },
    };
    let pose = params.transform.pose(0.0, SPF);
    fill_collider(
        &gpu,
        &mut cache,
        &params,
        &pose,
        2.0 / 32.0,
        ColliderFields {
            sdf: &out,
            velocity: &v,
        },
    )
    .unwrap();
    out.read_back(&gpu).unwrap()
}

/// With equal radii the cone is a cylinder along x, length 0.5, radius 0.2.
/// On its axis, inside, the distance is minus the nearer of the cap (0.25 − |x|)
/// and the side (0.2); outside the cap on the axis it is the gap to the cap.
#[test]
fn a_cylinder_cone_has_the_analytic_distance_on_its_axis() {
    let s = sdf(
        Shape::Cone {
            length: 0.5,
            radius_start: 0.2,
            radius_end: 0.2,
        },
        None,
        [1.0, 1.0, 1.0],
    );
    let cells = FieldDims::new(32, 32, 32);
    // Axis cells sit at y = z = 1.03125 (cell 16), 0.03125 m off the true axis,
    // so compare radially: distance to the side is 0.2 − 0.03125·√2.
    let off = (2.0f32 * 0.03125f32 * 0.03125).sqrt();
    let inside = s[index(cells, 16, 16, 16)]; // x = 1.03125, 0.03125 from the centre
    let expected_inside = -(0.2 - off).min(0.25 - 0.03125);
    assert!(
        (inside - expected_inside).abs() < 1e-5,
        "{inside} vs {expected_inside}"
    );
    // Cell 24 is x = 1.53125: 0.53125 from the centre, 0.28125 beyond the cap.
    let past = s[index(cells, 24, 16, 16)];
    let gap = 0.53125 - 0.25;
    assert!((past - gap).abs() < 1e-3, "{past} vs {gap}");
}

/// A true cone (radius_end 0) points at its tip: on the axis beyond the tip
/// the distance is the gap to the tip.
#[test]
fn a_true_cone_measures_to_its_tip() {
    let s = sdf(
        Shape::Cone {
            length: 0.5,
            radius_start: 0.2,
            radius_end: 0.0,
        },
        None,
        [1.0, 1.0, 1.0],
    );
    let cells = FieldDims::new(32, 32, 32);
    let past = s[index(cells, 24, 16, 16)]; // 0.28125 beyond the tip at 0.25
    // Axis cells are 0.03125 off in y and z, so the tip is √(0.28125² + 0.0442²) away.
    let expected = (0.28125f32.powi(2) + 2.0 * 0.03125f32.powi(2)).sqrt();
    assert!((past - expected).abs() < 2e-3, "{past} vs {expected}");
    assert!(s[index(cells, 16, 16, 16)] < 0.0, "the middle is inside");
}

/// `radius_start` is the -x end: a cone wide at -x and pointed at +x holds a
/// point 0.16 m off the axis near its wide end but not near its tip.
#[test]
fn radius_start_is_the_minus_x_end() {
    let s = sdf(
        Shape::Cone {
            length: 0.5,
            radius_start: 0.2,
            radius_end: 0.0,
        },
        None,
        [1.0, 1.0, 1.0],
    );
    let cells = FieldDims::new(32, 32, 32);
    // y is cell 18 (0.15625 off axis), z cell 16 (0.03125): 0.159 radially.
    // The radius is 0.1875 at x = -0.21875 (cell 12) and 0.0125 at +0.21875 (cell 19).
    assert!(s[index(cells, 12, 18, 16)] < 0.0, "the wide end holds it");
    assert!(s[index(cells, 19, 18, 16)] > 0.0, "the tip does not");
}

/// Past the wide -x cap, inside the wide radius, the distance is the gap to the
/// cap. Cell 10 is x = 0.65625, 0.34375 from the centre, 0.09375 past the cap at
/// -0.25; its radial offset 0.159 is under radius_start 0.2. The side term is
/// about 0.1025, so using radius_end (0) for the cap reads 0.1025, not 0.09375.
#[test]
fn the_wide_cap_measures_the_gap_to_the_cap() {
    let s = sdf(
        Shape::Cone {
            length: 0.5,
            radius_start: 0.2,
            radius_end: 0.0,
        },
        None,
        [1.0, 1.0, 1.0],
    );
    let cells = FieldDims::new(32, 32, 32);
    let d = s[index(cells, 10, 18, 16)];
    assert!((d - 0.09375).abs() < 1e-3, "{d} vs 0.09375");
}

/// A cone turned 90° about z points along +y.
#[test]
fn rotation_turns_the_cone() {
    let s = sdf(
        Shape::Cone {
            length: 0.8,
            radius_start: 0.1,
            radius_end: 0.1,
        },
        Some(Rotate {
            axis: [0.0, 0.0, 1.0],
            degrees: 90.0,
        }),
        [1.0, 1.0, 1.0],
    );
    let cells = FieldDims::new(32, 32, 32);
    // Cell 21 is y = 1.34375, 0.34375 from the centre, inside the half length 0.4
    // (cell 22 would be 0.40625, just past the end).
    assert!(s[index(cells, 16, 21, 16)] < 0.0, "along +y is inside");
    assert!(s[index(cells, 22, 16, 16)] > 0.0, "along +x is outside");
}

#[test]
fn bad_cones_are_rejected() {
    for (what, shape) in [
        (
            "zero length",
            Shape::Cone {
                length: 0.0,
                radius_start: 0.1,
                radius_end: 0.1,
            },
        ),
        (
            "negative radius",
            Shape::Cone {
                length: 1.0,
                radius_start: -0.1,
                radius_end: 0.1,
            },
        ),
        (
            "both radii zero",
            Shape::Cone {
                length: 1.0,
                radius_start: 0.0,
                radius_end: 0.0,
            },
        ),
        (
            "nan",
            Shape::Cone {
                length: f32::NAN,
                radius_start: 0.1,
                radius_end: 0.1,
            },
        ),
    ] {
        assert!(shape.validate("ember.emitter").is_err(), "{what}");
    }
    Shape::Cone {
        length: 1.0,
        radius_start: 0.0,
        radius_end: 0.1,
    }
    .validate("ember.emitter")
    .unwrap();
}
