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
    #[error("the GPU ran out of memory: {0}")]
    OutOfMemory(String),
    #[error("internal GPU error: {0}")]
    Internal(String),
}

/// The limits Elements asks a device for, given what its adapter supports.
///
/// Everything is `downlevel_defaults` except two things taken from the adapter.
/// The resolution limits, because `downlevel_defaults` caps 3D textures at 256,
/// which cannot hold a staggered face of a 256³ domain. And `max_buffer_size`,
/// because at 256 MiB it rejects any domain above about 406³, and read-back of
/// a 512³ field needs 512 MiB. Both are limits, not features, so
/// `required_features` stays empty and the portability guarantee is unchanged.
pub fn required_limits(adapter: &wgpu::Limits) -> wgpu::Limits {
    wgpu::Limits {
        max_buffer_size: adapter.max_buffer_size,
        ..wgpu::Limits::downlevel_defaults().using_resolution(adapter.clone())
    }
}

/// Owns the `wgpu` device and queue for one engine process.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    lost: Arc<Mutex<Option<String>>>,
    /// The first error raised outside any error scope, until `scoped` reports it.
    uncaptured: Arc<Mutex<Option<GpuError>>>,
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

        let required_limits = required_limits(&adapter.limits());

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

        // Without this, an error raised outside any error scope goes to wgpu's
        // default handler, which panics, and a daemon panic is exactly what
        // running the engine out of process exists to prevent. Keep the first
        // one; `scoped` reports it.
        let uncaptured = Arc::new(Mutex::new(None));
        let uncaptured_sink = Arc::clone(&uncaptured);
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            let mut slot = uncaptured_sink.lock().unwrap_or_else(|e| e.into_inner());
            if slot.is_none() {
                *slot = Some(from_wgpu(error));
            }
        }));

        Ok(Self {
            device,
            queue,
            adapter_name,
            lost,
            uncaptured,
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

    /// Run `f` inside out-of-memory, internal and validation error scopes,
    /// converting any captured error.
    ///
    /// This is the only sanctioned way to submit GPU work in Elements: it turns
    /// `wgpu`'s asynchronous, panicking-by-default error reporting into a
    /// `Result` the daemon can surface to the addon. See `resolve_errors` for
    /// which error wins when several are captured.
    ///
    /// # Limitations
    ///
    /// These scopes only catch encoding-time and descriptor errors during the
    /// execution of `f`. They do NOT catch device-timeline faults from work
    /// already submitted to the GPU: `Queue::submit` returns before the
    /// hardware runs the work, so faults that only manifest during execution
    /// may surface after `scoped` returns `Ok`, and may be misattributed to a
    /// later `scoped` block. Call sites that must ensure submitted work
    /// succeeded should wait for completion explicitly (for example via
    /// `Queue::on_submitted_work_done`) before trusting an `Ok`.
    ///
    /// An error raised outside any scope does not panic: the device's
    /// uncaptured-error handler holds the first one, and the next `scoped`
    /// call reports it, once. That call may be unrelated to its cause.
    pub fn scoped<T>(&self, f: impl FnOnce() -> T) -> Result<T, GpuError> {
        let out_of_memory = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal = self.device.push_error_scope(wgpu::ErrorFilter::Internal);
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let value = f();
        // Pop in reverse push order.
        let validation = pollster::block_on(validation.pop());
        let internal = pollster::block_on(internal.pop());
        let out_of_memory = pollster::block_on(out_of_memory.pop());
        let stray = self
            .uncaptured
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        match resolve_errors(
            self.device_lost(),
            out_of_memory,
            internal,
            validation,
            stray,
        ) {
            Some(e) => Err(e),
            None => Ok(value),
        }
    }
}

fn from_wgpu(error: wgpu::Error) -> GpuError {
    match error {
        wgpu::Error::OutOfMemory { .. } => GpuError::OutOfMemory(error.to_string()),
        wgpu::Error::Internal { .. } => GpuError::Internal(error.to_string()),
        wgpu::Error::Validation { .. } => GpuError::Validation(error.to_string()),
    }
}

/// What one `scoped` call reports when several things went wrong.
///
/// A lost device outranks everything, since nothing else can be retried, but
/// its message keeps the detail of whatever else was captured. Then out of
/// memory, which says why a step failed. Then internal errors, then
/// validation errors, then an error raised earlier outside any scope.
pub fn resolve_errors(
    lost: Option<String>,
    out_of_memory: Option<wgpu::Error>,
    internal: Option<wgpu::Error>,
    validation: Option<wgpu::Error>,
    stray: Option<GpuError>,
) -> Option<GpuError> {
    let captured = out_of_memory
        .map(from_wgpu)
        .or_else(|| internal.map(from_wgpu))
        .or_else(|| validation.map(from_wgpu))
        .or(stray);
    match (lost, captured) {
        (Some(message), Some(e)) => Some(GpuError::DeviceLost(format!(
            "{message} (another error was also captured: {e})"
        ))),
        (Some(message), None) => Some(GpuError::DeviceLost(message)),
        (None, captured) => captured,
    }
}
