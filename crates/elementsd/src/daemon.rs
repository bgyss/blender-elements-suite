//! Session state and command handling.
//!
//! Every handler is total: a failure becomes a `Response::Error` and the
//! session stays alive. The daemon outliving a bad graph is the whole point of
//! running out of process.

use std::path::Path;

use elements_core::gpu::{FieldDims, FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{Document, Graph, NodeRegistry};
use elements_ipc::{
    Command, ELEMENTS_PROTOCOL_VERSION, EngineError, ErrorKind, FrameWriter, Response,
};

/// One client's engine state.
pub struct Session {
    gpu: GpuContext,
    pool: FieldPool,
    pipelines: PipelineCache,
    registry: NodeRegistry,
    graph: Option<(Graph, FieldDims)>,
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
            registry: NodeRegistry::with_builtins(),
            graph: None,
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

        Command::Render { frame: _ } => {
            if let Err(e) = require_greeted(session) {
                return Response::Error(e);
            }
            match render(session) {
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
    let (graph, dims) = doc
        .into_graph(&session.registry)
        .map_err(|e| EngineError::new(ErrorKind::Document, e.to_string()))?;

    // `GpuContext` requests `Limits::downlevel_defaults()`, which caps
    // `max_texture_dimension_3d` well below what some adapters could
    // actually support -- 256, on a downlevel profile. Without this check, a
    // document whose dims exceed that limit answers `Loaded` here (nothing
    // above touches the GPU) and only fails later, on every render, as a
    // generic `ErrorKind::Gpu` -- which the add-on treats as transient and
    // retries, when the document can in fact never render. Read the limit
    // from the device actually in use rather than hardcoding the 256 default,
    // so this keeps working if the requested limits ever change.
    let max_dim = session.gpu.device().limits().max_texture_dimension_3d;
    for (axis, value) in [("x", dims.x), ("y", dims.y), ("z", dims.z)] {
        if value > max_dim {
            return Err(EngineError::new(
                ErrorKind::Document,
                format!(
                    "dims {:?}: {axis} = {value} exceeds this device's max_texture_dimension_3d \
                     of {max_dim}",
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

    session.graph = Some((graph, dims));
    Ok(Response::Loaded {
        dims: [dims.x, dims.y, dims.z],
        nodes,
    })
}

fn render(session: &mut Session) -> Result<Response, EngineError> {
    let (graph, dims) = session
        .graph
        .as_ref()
        .ok_or_else(|| EngineError::new(ErrorKind::Graph, "no graph is loaded"))?;

    let value = graph
        .eval(
            &session.gpu,
            &mut session.pool,
            &mut session.pipelines,
            *dims,
        )
        .map_err(map_node_error)?;
    let field = value.as_field().map_err(map_node_error)?;
    let values = field.read_back(&session.gpu).map_err(|e| {
        let kind = if session.gpu.device_lost().is_some() {
            ErrorKind::DeviceLost
        } else {
            ErrorKind::Gpu
        };
        EngineError::new(kind, e.to_string())
    })?;

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
