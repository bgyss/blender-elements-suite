use elements_ember::transform::{Key, Rotate, Shape, Transform};

const SPF: f64 = 1.0 / 24.0;

fn key(frame: f32, translate: [f32; 3], rotate: Option<([f32; 3], f32)>) -> Key {
    Key {
        frame,
        translate,
        rotate: rotate.map(|(axis, degrees)| Rotate { axis, degrees }),
    }
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol, "{what}: {a} vs {b}");
}

/// Spec §2.1: translation is linear between keys, and velocity is the
/// segment's slope in m/s.
#[test]
fn translation_interpolates_linearly_and_velocity_is_its_slope() {
    let t = Transform {
        keys: vec![key(0.0, [0.0; 3], None), key(10.0, [1.0, 2.0, 0.0], None)],
    };
    let p = t.pose(5.0, SPF);
    close(p.translate[0], 0.5, 1e-12, "x");
    close(p.translate[1], 1.0, 1e-12, "y");
    let seconds = 10.0 * SPF;
    close(p.linear[0], 1.0 / seconds, 1e-9, "vx");
    close(p.linear[1], 2.0 / seconds, 1e-9, "vy");
}

/// Rotation is a slerp: a quarter of the way through a 170° turn is 42.5°.
/// A normalised linear blend of quaternions lands measurably elsewhere.
#[test]
fn rotation_interpolates_by_slerp_with_constant_angular_velocity() {
    let t = Transform {
        keys: vec![
            key(0.0, [0.0; 3], None),
            key(8.0, [0.0; 3], Some(([0.0, 0.0, 1.0], 170.0))),
        ],
    };
    let p = t.pose(2.0, SPF);
    close(
        p.angle_about([0.0, 0.0, 1.0]),
        42.5f64.to_radians(),
        1e-6,
        "angle",
    );
    let omega = 170f64.to_radians() / (8.0 * SPF);
    close(p.angular[2], omega, 1e-6, "ωz");
    close(p.angular[0], 0.0, 1e-9, "ωx");
}

#[test]
fn a_transform_holds_before_its_first_key_and_after_its_last() {
    let t = Transform {
        keys: vec![
            key(5.0, [1.0, 0.0, 0.0], None),
            key(9.0, [3.0, 0.0, 0.0], None),
        ],
    };
    for (frame, x) in [(0.0, 1.0), (5.0, 1.0), (9.0, 3.0), (30.0, 3.0)] {
        let p = t.pose(frame, SPF);
        close(p.translate[0], x, 1e-12, &format!("x at {frame}"));
        if !(5.0..9.0).contains(&frame) {
            assert_eq!(p.linear, [0.0; 3], "still at {frame}");
        }
    }
}

/// v(x) = linear + ω × (x − translate).
#[test]
fn material_velocity_adds_the_spin_about_the_origin() {
    let t = Transform {
        keys: vec![
            key(0.0, [1.0, 0.0, 0.0], None),
            key(24.0, [1.0, 0.0, 0.0], Some(([0.0, 0.0, 1.0], 90.0))),
        ],
    };
    let p = t.pose(12.0, SPF);
    let omega = std::f64::consts::FRAC_PI_2 / 1.0; // 90° over 24 frames = 1 s
    let v = p.velocity_at([1.0, 1.0, 0.0]);
    close(v[0], -omega, 1e-9, "vx");
    close(v[1], 0.0, 1e-9, "vy");
}

fn rejected_transform(keys: Vec<Key>) -> bool {
    Transform { keys }.validate("test").is_err()
}

#[test]
fn bad_shapes_and_transforms_are_rejected() {
    assert!(rejected_transform(vec![]));
    assert!(rejected_transform(vec![
        key(3.0, [0.0; 3], None),
        key(3.0, [0.0; 3], None)
    ]));
    assert!(rejected_transform(vec![key(
        0.0,
        [f32::NAN, 0.0, 0.0],
        None
    )]));
    assert!(rejected_transform(vec![key(
        0.0,
        [0.0; 3],
        Some(([0.0; 3], 10.0))
    )]));
    assert!(!rejected_transform(vec![key(
        0.0,
        [0.0; 3],
        Some(([0.0, 1.0, 0.0], 10.0))
    )]));
    assert!(Shape::Sphere { radius: 0.0 }.validate("test").is_err());
    assert!(
        Shape::Box {
            half_extents: [0.1, -0.1, 0.1]
        }
        .validate("test")
        .is_err()
    );
    assert!(
        Shape::Box {
            half_extents: [0.1, 0.1, 0.1]
        }
        .validate("test")
        .is_ok()
    );
}

#[test]
fn shapes_parse_from_documents() {
    let s: Shape =
        serde_json::from_value(serde_json::json!({ "box": { "half_extents": [1.0, 2.0, 3.0] } }))
            .unwrap();
    assert_eq!(
        s,
        Shape::Box {
            half_extents: [1.0, 2.0, 3.0]
        }
    );
    let t: Transform = serde_json::from_value(serde_json::json!({
        "keys": [{ "frame": 1, "translate": [1, 0, 0], "rotate": { "axis": [0, 0, 1], "degrees": 45 } }]
    }))
    .unwrap();
    assert_eq!(t.keys.len(), 1);
    assert!(serde_json::from_value::<Shape>(serde_json::json!({ "cone": {} })).is_err());
}
