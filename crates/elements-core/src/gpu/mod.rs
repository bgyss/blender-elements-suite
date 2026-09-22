//! GPU device acquisition and error-scope handling.

mod batch;
mod dispatch;
mod field;
mod pool;
mod staggered;

pub use batch::ComputeBatch;
pub use dispatch::{
    PipelineCache, WORKGROUP, accumulate_into, dispatch_over_field, fill_constant, fill_curl_noise,
};
pub use field::{Field, FieldDims, FieldFormat, validate_dims_fit_buffer_limit};
pub use pool::FieldPool;
pub use staggered::{Axis, StaggeredField};

use std::sync::{Arc, Mutex};

/// Every way GPU work can fail in Elements.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("no suitable GPU adapter was found")]
    NoAdapter,
    #[error("could not create a GPU device: {0}")]
    DeviceRequest(String),
    #[error("GPU validation error: {0}")]
    Validation(String),
    #[error("GPU device was lost: {0}")]
    DeviceLost(String),
}

/// Owns the `wgpu` device and queue for one engine process.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    lost: Arc<Mutex<Option<String>>>,
}

impl GpuContext {
    /// Acquire a device with no surface, suitable for daemons, CLI bakes and CI.
    pub fn new_headless() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_headless_async())
    }

    async fn new_headless_async() -> Result<Self, GpuError> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|_| GpuError::NoAdapter)?;

        let adapter_name = adapter.get_info().name;

        let required_features = wgpu::Features::empty();

        // Resolution limits come from the adapter. `downlevel_defaults` caps
        // 3D textures at 256, which cannot hold a staggered face (n + 1 cells)
        // of a 256³ domain, let alone a 512³ bake. These are limits, not
        // features, so `required_features` stays empty and the portability
        // guarantee is unchanged.
        let required_limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("elements-device"),
                required_features,
                required_limits,
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuError::DeviceRequest(e.to_string()))?;

        let lost = Arc::new(Mutex::new(None));
        let lost_sink = Arc::clone(&lost);
        device.set_device_lost_callback(move |_reason, message| {
            *lost_sink.lock().unwrap_or_else(|e| e.into_inner()) = Some(message);
        });

        Ok(Self {
            device,
            queue,
            adapter_name,
            lost,
        })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Returns the device-lost message if the device has been lost.
    pub fn device_lost(&self) -> Option<String> {
        self.lost.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Run `f` inside a validation error scope, converting any captured error.
    ///
    /// This is the only sanctioned way to submit GPU work in Elements: it turns
    /// `wgpu`'s asynchronous, panicking-by-default error reporting into a
    /// `Result` the daemon can surface to the addon.
    ///
    /// # Limitations
    ///
    /// This scope only catches encoding-time and descriptor validation errors
    /// during the execution of `f`. It does NOT catch device-timeline faults
    /// from work already submitted to the GPU: `Queue::submit` returns before
    /// the hardware runs the work, so faults that only manifest during execution
    /// may surface after `scoped` returns `Ok`, and may be misattributed to a
    /// later `scoped` block. Call sites that must ensure submitted work succeeded
    /// should wait for completion explicitly (for example via
    /// `Queue::on_submitted_work_done`) before trusting an `Ok`.
    pub fn scoped<T>(&self, f: impl FnOnce() -> T) -> Result<T, GpuError> {
        let guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let value = f();
        let error = pollster::block_on(guard.pop());

        if let Some(message) = self.device_lost() {
            if let Some(e) = error {
                return Err(GpuError::DeviceLost(format!(
                    "{message} (a validation error was also captured: {e})"
                )));
            }
            return Err(GpuError::DeviceLost(message));
        }
        match error {
            Some(e) => Err(GpuError::Validation(e.to_string())),
            None => Ok(value),
        }
    }
}
