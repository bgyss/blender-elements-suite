use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, GpuContext, GpuError};

#[test]
fn acquires_a_headless_device() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    assert!(!ctx.adapter_name().is_empty());
}

#[test]
fn scoped_reports_validation_errors() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");

    // A zero-size buffer with MAP_READ did not trip validation on this wgpu/Metal
    // combination (wgpu 30.0.1); a texture with width 0 is the brief's sanctioned
    // fallback invalid resource, and it does route through the error scope here.
    let result = ctx.scoped(|| {
        let _ = ctx.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("deliberately-invalid"),
            size: wgpu::Extent3d {
                width: 0,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
    });

    match result {
        Err(GpuError::Validation(msg)) => assert!(!msg.is_empty()),
        other => panic!("expected a validation error, got {other:?}"),
    }
}

#[test]
fn scoped_passes_through_success() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let value = ctx.scoped(|| 41 + 1).expect("valid work should not error");
    assert_eq!(value, 42);
}

/// A staggered velocity face of a 256³ domain is 257 cells along its own axis.
/// `Limits::downlevel_defaults()` caps 3D textures at 256, so this allocation
/// is the capability Ember needs, tested directly.
#[test]
fn a_staggered_face_of_a_256_domain_can_be_allocated() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let mut pool = FieldPool::new();
    let face = pool.acquire(&ctx, FieldDims::new(257, 4, 4), FieldFormat::R32Float);
    assert!(
        face.is_ok(),
        "257-wide 3D texture was refused: {:?}",
        face.err()
    );
}
