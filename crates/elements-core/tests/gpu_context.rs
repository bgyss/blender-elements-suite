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

/// Only the resolution limits and `max_buffer_size` may come from the adapter.
/// Everything else stays at `downlevel_defaults`, so kernels keep running on
/// every backend, in particular within 4 storage textures per stage.
#[test]
fn required_limits_take_only_resolution_and_buffer_size_from_the_adapter() {
    let mut adapter = wgpu::Limits::downlevel_defaults();
    adapter.max_buffer_size = 4 << 30;
    adapter.max_texture_dimension_3d = 2048;
    adapter.max_storage_textures_per_shader_stage = 16;

    let got = elements_core::gpu::required_limits(&adapter);
    let base = wgpu::Limits::downlevel_defaults();
    assert_eq!(got.max_buffer_size, 4 << 30);
    assert_eq!(got.max_texture_dimension_3d, 2048);
    assert_eq!(
        got.max_storage_textures_per_shader_stage,
        base.max_storage_textures_per_shader_stage
    );
}

fn source() -> wgpu::ErrorSource {
    Box::new(std::io::Error::other("test"))
}

fn oom() -> wgpu::Error {
    wgpu::Error::OutOfMemory { source: source() }
}

fn validation() -> wgpu::Error {
    wgpu::Error::Validation {
        source: source(),
        description: "bad".into(),
    }
}

fn internal() -> wgpu::Error {
    wgpu::Error::Internal {
        source: source(),
        description: "internal".into(),
    }
}

/// An out-of-memory error must reach the caller as `OutOfMemory`, not be
/// folded into `Validation`, and must win over a validation error captured
/// alongside it: it is the one that says why.
#[test]
fn out_of_memory_is_reported_as_itself_and_outranks_validation() {
    use elements_core::gpu::resolve_errors;
    assert!(matches!(
        resolve_errors(None, Some(oom()), None, None, None),
        Some(GpuError::OutOfMemory(_))
    ));
    assert!(matches!(
        resolve_errors(None, Some(oom()), None, Some(validation()), None),
        Some(GpuError::OutOfMemory(_))
    ));
    assert!(matches!(
        resolve_errors(Some("gone".into()), Some(oom()), None, None, None),
        Some(GpuError::DeviceLost(_))
    ));
    assert!(matches!(
        resolve_errors(
            None,
            None,
            None,
            None,
            Some(GpuError::Validation("earlier".into()))
        ),
        Some(GpuError::Validation(_))
    ));
    assert!(resolve_errors(None, None, None, None, None).is_none());
}

/// An internal error is reported as itself and outranks a validation error
/// captured alongside it. A lost device keeps the detail of whatever else
/// was captured, wrapped into its own message.
#[test]
fn internal_outranks_validation_and_device_lost_keeps_the_other_detail() {
    use elements_core::gpu::resolve_errors;
    assert!(matches!(
        resolve_errors(None, None, Some(internal()), Some(validation()), None),
        Some(GpuError::Internal(_))
    ));
    match resolve_errors(Some("gone".into()), Some(oom()), None, None, None) {
        Some(GpuError::DeviceLost(msg)) => {
            assert!(msg.contains("gone"), "{msg}");
            assert!(msg.contains("Out of Memory"), "{msg}");
        }
        other => panic!("expected DeviceLost, got {other:?}"),
    }
}

/// A validation error raised outside any error scope used to reach wgpu's
/// default handler, which panics. It must instead be reported by the next
/// `scoped` call, and only once.
#[test]
fn an_error_outside_any_scope_is_reported_later_not_panicked() {
    let ctx = GpuContext::new_headless().expect("no GPU adapter available");
    let _ = ctx.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("deliberately-invalid-outside-a-scope"),
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
    match ctx.scoped(|| ()) {
        Err(GpuError::Validation(msg)) => assert!(!msg.is_empty()),
        other => panic!("expected the stray validation error, got {other:?}"),
    }
    assert!(
        ctx.scoped(|| ()).is_ok(),
        "a stray error is reported once, not forever"
    );
}
