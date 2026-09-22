//! Session state and command handling.
//!
//! Every handler is total: a failure becomes a `Response::Error` and the
//! session stays alive. The daemon outliving a bad graph is the whole point of
//! running out of process.

use std::path::Path;

use elements_core::gpu::{FieldDims, FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{Document, Graph, NodeRegistry, Timeline};
use elements_ipc::{
    Command, ELEMENTS_PROTOCOL_VERSION, EngineError, ErrorKind, FrameWriter, Response,
};

/// The most a requested frame may exceed the timeline's `start_frame` by.
///
/// `Timeline::goto` on a stateful graph steps forward one evaluation per
/// frame between the held/cached state and the target, so an unbounded
/// `Render { frame }` from untrusted IPC (`frame: u32::MAX` is about 4.3e9)
/// hangs the daemon evaluating billions of frames. Frames below
/// `start_frame` are unaffected: `Timeline::goto` clamps them, it never
/// steps for them. Value matches Blender's own `MAXFRAME`, so any frame a
/// real Blender timeline could name is always allowed.
const MAX_FRAME_SPAN: u64 = 1_048_574;

/// One client's engine state.
pub struct Session {
    gpu: GpuContext,
    pool: FieldPool,
    pipelines: PipelineCache,
    registry: NodeRegistry,
    graph: Option<(Graph, FieldDims)>,
    /// Time and simulation state for the loaded graph.
    timeline: Option<Timeline>,
    channel_path: std::path::PathBuf,
    writer: Option<FrameWriter>,
    greeted: bool,
}

impl Session {
    pub fn new(channel_path: &Path) -> Result<Self, EngineError> {
        let gpu = GpuContext::new_headless()
            .map_err(|e| EngineError::new(ErrorKind::Gpu, e.to_string()))?;
        Ok(Self {
            gpu,
            pool: FieldPool::new(),
            pipelines: PipelineCache::new(),
            registry: elements_ember::registry(),
            graph: None,
            timeline: None,
            channel_path: channel_path.to_path_buf(),
            writer: None,
            greeted: false,
        })
    }

    pub fn adapter_name(&self) -> &str {
        self.gpu.adapter_name()
    }
}

/// Handle one command. Never panics; never returns `Err`.
pub fn handle(session: &mut Session, command: Command) -> Response {
    match command {
        Command::Hello { protocol_version } => {
            if protocol_version != ELEMENTS_PROTOCOL_VERSION {
                return Response::Error(EngineError::new(
                    ErrorKind::ProtocolVersion,
                    format!(
                        "client speaks protocol {protocol_version}, this engine speaks \
                         {ELEMENTS_PROTOCOL_VERSION}; reinstall the add-on"
                    ),
                ));
            }
            session.greeted = true;
            Response::HelloAck {
                protocol_version: ELEMENTS_PROTOCOL_VERSION,
                engine_version: env!("CARGO_PKG_VERSION").to_owned(),
                adapter: session.adapter_name().to_owned(),
            }
        }

        Command::LoadGraph { path } => {
            if let Err(e) = require_greeted(session) {
                return Response::Error(e);
            }
            match load(session, Path::new(&path)) {
                Ok(response) => response,
                Err(e) => Response::Error(e),
            }
        }

        Command::Render { frame } => {
            if let Err(e) = require_greeted(session) {
                return Response::Error(e);
            }
            match render(session, frame) {
                Ok(response) => response,
                Err(e) => Response::Error(e),
            }
        }

        Command::Shutdown => Response::Bye,
    }
}

fn require_greeted(session: &Session) -> Result<(), EngineError> {
    if session.greeted {
        Ok(())
    } else {
        Err(EngineError::new(
            ErrorKind::ProtocolVersion,
            "client must send Hello first",
        ))
    }
}

fn load(session: &mut Session, path: &Path) -> Result<Response, EngineError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| EngineError::new(ErrorKind::Io, format!("{}: {e}", path.display())))?;
    let doc = Document::from_json(&text)
        .map_err(|e| EngineError::new(ErrorKind::Document, e.to_string()))?;

    // Reject a document whose implied fields would not fit this device's
    // `max_buffer_size`, before the frame channel or any GPU texture is
    // allocated. This is a distinct check from the `max_texture_dimension_3d`
    // loop below: `max_buffer_size` (256 MiB) is reachable well inside the
    // adapter's texture-dimension limit (2048 on Apple Silicon), since a
    // padded R32Float readback of a domain far under 2048^3 is already
    // gigabytes. See `Document::validate_for`'s doc comment.
    doc.validate_for(&session.gpu.device().limits())
        .map_err(|e| EngineError::new(ErrorKind::Document, e.to_string()))?;

    let config = doc.timeline_config();
    let (graph, dims) = doc
        .into_graph(&session.registry)
        .map_err(|e| EngineError::new(ErrorKind::Document, e.to_string()))?;

    // `GpuContext` requests the adapter's own resolution limits, so this is
    // the real hardware cap (2048 on Apple Silicon). Without this check, a
    // document whose dims exceed it answers `Loaded` here (nothing above
    // touches the GPU) and only fails later, on every render, as a generic
    // `ErrorKind::Gpu`, which the add-on treats as transient and retries.
    //
    // The comparison is `>=`, not `>`: staggered vector fields store one more
    // face than cells along each axis, so a domain axis must stay strictly
    // below the limit for its faces to fit.
    let max_dim = session.gpu.device().limits().max_texture_dimension_3d;
    for (axis, value) in [("x", dims.x), ("y", dims.y), ("z", dims.z)] {
        if value >= max_dim {
            return Err(EngineError::new(
                ErrorKind::Document,
                format!(
                    "dims {:?}: {axis} = {value} must be below this device's \
                     max_texture_dimension_3d of {max_dim} (staggered faces need one extra cell)",
                    [dims.x, dims.y, dims.z]
                ),
            ));
        }
    }

    let nodes = graph.node_count() as u32;

    // Reallocate the channel whenever the resolution changes.
    let needs_channel = match &session.writer {
        Some(w) => w.dims() != [dims.x, dims.y, dims.z],
        None => true,
    };
    if needs_channel {
        session.writer = Some(
            FrameWriter::create(&session.channel_path, [dims.x, dims.y, dims.z], 1).map_err(
                |e| match e {
                    elements_ipc::ChannelError::FieldTooLarge { .. } => {
                        EngineError::new(ErrorKind::Document, e.to_string())
                    }
                    other => EngineError::new(ErrorKind::Io, other.to_string()),
                },
            )?,
        );
    }

    // A new document invalidates every frame the old one simulated. Its
    // textures go back to the pool, since the device is still good.
    if let Some(mut old) = session.timeline.take() {
        old.reset(&mut session.pool);
    }
    session.graph = Some((graph, dims));
    session.timeline = Some(Timeline::new(config));
    Ok(Response::Loaded {
        dims: [dims.x, dims.y, dims.z],
        nodes,
    })
}

fn render(session: &mut Session, frame: u32) -> Result<Response, EngineError> {
    let no_graph = || EngineError::new(ErrorKind::Graph, "no graph is loaded");
    let (graph, dims) = session.graph.as_ref().ok_or_else(no_graph)?;
    let dims = *dims;
    let timeline = session.timeline.as_mut().ok_or_else(no_graph)?;

    // Reject a frame far past `start_frame` before it ever reaches
    // `Timeline::goto`, which would otherwise step forward one evaluation per
    // frame in between. Frames below `start_frame` are unaffected: `goto`
    // clamps them rather than stepping. Saturating/`u64` throughout so no
    // `u32` value from IPC can overflow this check.
    let start_frame = timeline.config().start_frame as u64;
    let span = (frame as u64).saturating_sub(start_frame);
    if span > MAX_FRAME_SPAN {
        return Err(EngineError::new(
            ErrorKind::Graph,
            format!(
                "requested frame {frame} is {span} frames past the timeline's start frame \
                 {start_frame}, more than the {MAX_FRAME_SPAN}-frame limit"
            ),
        ));
    }

    let evaluated = match timeline.goto(
        graph,
        &session.gpu,
        &mut session.pool,
        &mut session.pipelines,
        dims,
        frame,
    ) {
        Ok(evaluated) => evaluated,
        Err(e) => {
            let err = map_node_error(e);
            if err.kind == ErrorKind::DeviceLost {
                timeline.discard();
            }
            return Err(err);
        }
    };
    if let Some(warning) = timeline.take_warning() {
        // stderr, never stdout: stdout carries the readiness line.
        eprintln!("elementsd: warning: {warning}");
    }

    let values = match evaluated.value.as_field() {
        Ok(field) => field.read_back(&session.gpu).map_err(|e| {
            let kind = if session.gpu.device_lost().is_some() {
                ErrorKind::DeviceLost
            } else {
                ErrorKind::Gpu
            };
            EngineError::new(kind, e.to_string())
        }),
        Err(e) => Err(map_node_error(e)),
    };
    evaluated.value.release_to(&mut session.pool);
    let values = match values {
        Ok(values) => values,
        Err(e) => {
            if e.kind == ErrorKind::DeviceLost {
                timeline.discard();
            }
            return Err(e);
        }
    };

    let writer = session
        .writer
        .as_mut()
        .ok_or_else(|| EngineError::new(ErrorKind::Io, "no frame channel"))?;
    let seq = writer
        .publish(&values)
        .map_err(|e| EngineError::new(ErrorKind::Io, e.to_string()))?;

    Ok(Response::Frame {
        seq,
        channel: session.channel_path.to_string_lossy().into_owned(),
        dims: [dims.x, dims.y, dims.z],
    })
}

fn map_node_error(e: elements_core::graph::NodeError) -> EngineError {
    use elements_core::gpu::GpuError;
    use elements_core::graph::NodeError;
    let kind = match &e {
        NodeError::Gpu(GpuError::DeviceLost(_)) => ErrorKind::DeviceLost,
        NodeError::Gpu(_) => ErrorKind::Gpu,
        _ => ErrorKind::Graph,
    };
    EngineError::new(kind, e.to_string())
}

#[cfg(test)]
mod tests {
    use super::map_node_error;
    use elements_core::gpu::GpuError;
    use elements_core::graph::NodeError;
    use elements_ipc::ErrorKind;

    #[test]
    fn device_lost_maps_to_device_lost_not_a_generic_gpu_error() {
        let e = NodeError::Gpu(GpuError::DeviceLost("test".into()));
        let mapped = map_node_error(e);
        assert_eq!(mapped.kind, ErrorKind::DeviceLost);
    }
}
