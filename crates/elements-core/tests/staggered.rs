use elements_core::gpu::{
    Axis, FieldDims, FieldFormat, FieldPool, GpuContext, GpuError, PipelineCache, StaggeredField,
};
use elements_core::graph::{NodeError, SocketType, Value};

fn gpu() -> GpuContext {
    GpuContext::new_headless().expect("no GPU adapter available")
}

#[test]
fn face_dims_add_one_cell_along_their_own_axis() {
    let cells = FieldDims::new(4, 5, 6);
    assert_eq!(
        StaggeredField::face_dims(cells, Axis::X),
        FieldDims::new(5, 5, 6)
    );
    assert_eq!(
        StaggeredField::face_dims(cells, Axis::Y),
        FieldDims::new(4, 6, 6)
    );
    assert_eq!(
        StaggeredField::face_dims(cells, Axis::Z),
        FieldDims::new(4, 5, 7)
    );
}

#[test]
fn acquire_staggered_zeroed_gives_three_zeroed_faces_of_the_right_shape() {
    let ctx = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(4, 5, 6);

    let field = pool
        .acquire_staggered_zeroed(&ctx, &mut cache, cells)
        .unwrap();
    assert_eq!(field.cells(), cells);
    for axis in Axis::ALL {
        let face = field.face(axis);
        assert_eq!(face.dims(), StaggeredField::face_dims(cells, axis));
        assert_eq!(face.format(), FieldFormat::R32Float);
        assert!(face.read_back(&ctx).unwrap().iter().all(|&v| v == 0.0));
    }
}

#[test]
fn from_faces_rejects_a_face_of_the_wrong_shape() {
    let ctx = gpu();
    let mut pool = FieldPool::new();
    let cells = FieldDims::new(4, 5, 6);

    // The x face is given cell dims instead of (nx + 1, ny, nz).
    let x = pool.acquire(&ctx, cells, FieldFormat::R32Float).unwrap();
    let y = pool
        .acquire(
            &ctx,
            StaggeredField::face_dims(cells, Axis::Y),
            FieldFormat::R32Float,
        )
        .unwrap();
    let z = pool
        .acquire(
            &ctx,
            StaggeredField::face_dims(cells, Axis::Z),
            FieldFormat::R32Float,
        )
        .unwrap();

    match StaggeredField::from_faces(cells, [x, y, z]) {
        Err(GpuError::Validation(message)) => assert!(message.contains("X"), "{message}"),
        other => panic!("expected a Validation error, got {other:?}"),
    }
}

#[test]
fn releasing_a_staggered_field_returns_all_three_faces() {
    let ctx = gpu();
    let mut pool = FieldPool::new();
    let field = pool
        .acquire_staggered_uninit(&ctx, FieldDims::new(4, 4, 4))
        .unwrap();
    assert_eq!(pool.pooled_count(), 0);
    Value::VectorField(field).release_to(&mut pool);
    assert_eq!(pool.pooled_count(), 3);
}

#[test]
fn vector_values_are_typed() {
    let ctx = gpu();
    let mut pool = FieldPool::new();
    let field = pool
        .acquire_staggered_uninit(&ctx, FieldDims::new(2, 2, 2))
        .unwrap();
    let value = Value::VectorField(field);

    assert_eq!(value.socket_type(), SocketType::VectorField);
    assert!(value.as_vector_field().is_ok());
    assert!(matches!(
        value.as_field(),
        Err(NodeError::TypeMismatch {
            expected: SocketType::Field,
            ..
        })
    ));
    assert!(matches!(
        Value::Scalar(1.0).as_vector_field(),
        Err(NodeError::TypeMismatch {
            expected: SocketType::VectorField,
            ..
        })
    ));
}
