use elements_ember::mesh::{MAX_TRIANGLES, MAX_VERTICES, Mesh};

fn outward_volume(m: &Mesh) -> f64 {
    // Signed volume by the divergence theorem: positive for outward winding.
    m.indices
        .chunks(3)
        .map(|t| {
            let [a, b, c] = [0, 1, 2].map(|i| m.positions[t[i] as usize].map(f64::from));
            (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                / 6.0
        })
        .sum()
}

#[test]
fn a_box_mesh_is_closed_and_wound_outward() {
    let m = Mesh::box_mesh([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]);
    assert_eq!(m.triangle_count(), 12);
    assert!(
        (outward_volume(&m) - 6.0).abs() < 1e-9,
        "volume {}",
        outward_volume(&m)
    );
    m.validate("test").unwrap();
}

#[test]
fn merging_offsets_the_second_meshs_indices() {
    let mut a = Mesh::box_mesh([0.0; 3], [1.0; 3]);
    let b = Mesh::box_mesh([2.0; 3], [3.0; 3]);
    a.merge(&b);
    assert_eq!(a.positions.len(), 16);
    assert_eq!(a.triangle_count(), 24);
    assert!(a.indices[36..].iter().all(|&i| i >= 8));
    assert!((outward_volume(&a) - 2.0).abs() < 1e-9);
}

#[test]
fn bad_meshes_are_rejected_with_a_reason() {
    let ok = Mesh::box_mesh([0.0; 3], [1.0; 3]);
    let cases: Vec<(&str, Mesh)> = vec![
        (
            "no triangles",
            Mesh {
                positions: vec![],
                indices: vec![],
            },
        ),
        (
            "not a multiple of 3",
            Mesh {
                indices: vec![0, 1],
                ..ok.clone()
            },
        ),
        (
            "out of range",
            Mesh {
                indices: vec![0, 1, 8],
                ..ok.clone()
            },
        ),
        (
            "must be finite",
            Mesh {
                positions: {
                    let mut p = ok.positions.clone();
                    p[3][1] = f32::NAN;
                    p
                },
                ..ok.clone()
            },
        ),
        (
            "at most",
            Mesh {
                positions: ok.positions.clone(),
                indices: vec![0; (MAX_TRIANGLES + 1) * 3],
            },
        ),
        (
            "positions, at most",
            Mesh {
                positions: vec![[0.0; 3]; MAX_VERTICES + 1],
                indices: vec![0, 1, 2],
            },
        ),
    ];
    for (what, m) in cases {
        let e = m.validate("ember.mesh_collider").unwrap_err().to_string();
        assert!(e.contains("ember.mesh_collider"), "{what}: {e}");
        // `what` is the reason text the rejection must carry.
        assert!(e.contains(what), "{what}: {e}");
    }
}

#[test]
fn obj_text_loads_with_slashed_indices_and_polygons() {
    let obj = "# a quad and a triangle\n\
               v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0 0 1\n\
               vt 0 0\nvn 0 0 1\n\
               f 1/1/1 2/1/1 3/1/1 4/1/1\n\
               f 1 2 5\n";
    let m = Mesh::from_obj(obj).unwrap();
    assert_eq!(m.positions.len(), 5);
    assert_eq!(m.triangle_count(), 3, "the quad fans into two triangles");
    assert_eq!(&m.indices[..6], &[0, 1, 2, 0, 2, 3]);
    assert!(
        Mesh::from_obj("f 1 2 3\n").is_err(),
        "indices need vertices"
    );
    assert!(
        Mesh::from_obj("v 0 0 0\nf -1 -1 -1\n").is_err(),
        "negative indices unsupported"
    );
}

#[test]
fn bounds_are_the_axis_aligned_extent() {
    let m = Mesh::box_mesh([-1.0, 0.5, 2.0], [3.0, 1.5, 4.0]);
    assert_eq!(m.bounds(), ([-1.0, 0.5, 2.0], [3.0, 1.5, 4.0]));
}
