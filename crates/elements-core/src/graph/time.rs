//! Simulation time as nodes see it.

/// The frame rate a document gets when it does not specify one.
pub const DEFAULT_FPS: f64 = 24.0;

/// The first frame of a timeline when a document does not specify one.
/// Matches Blender's default scene start.
pub const DEFAULT_START_FRAME: u32 = 1;

/// When a node is being evaluated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Time {
    /// The frame being produced.
    pub frame: u32,
    /// Seconds elapsed since the timeline's start frame.
    pub seconds: f64,
    /// Seconds per frame. Substepping is a solver's own business.
    pub dt: f64,
}

impl Time {
    /// Time at `frame` on a timeline that starts at `start_frame` and runs at `fps`.
    ///
    /// Frames before `start_frame` report zero seconds. The timeline clamps
    /// them to the start frame anyway.
    pub fn at(frame: u32, start_frame: u32, fps: f64) -> Self {
        Self {
            frame,
            seconds: frame.saturating_sub(start_frame) as f64 / fps,
            dt: 1.0 / fps,
        }
    }
}
