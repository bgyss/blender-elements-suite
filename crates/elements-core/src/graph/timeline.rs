//! Stepping a stateful graph through time, with a cache of frame snapshots.

use std::collections::BTreeMap;

use crate::gpu::{FieldDims, FieldPool, GpuContext, PipelineCache};

use super::node::NodeError;
use super::state::{Snapshot, StateStore};
use super::time::{DEFAULT_FPS, DEFAULT_START_FRAME, Time};
use super::{Evaluated, Graph};

/// The frame-cache budget a document gets when it does not specify one.
pub const DEFAULT_CACHE_BUDGET_MB: u32 = 2048;

/// How one timeline runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimelineConfig {
    pub fps: f64,
    pub start_frame: u32,
    /// GPU memory the snapshot cache may hold.
    pub cache_budget_bytes: u64,
}

impl Default for TimelineConfig {
    fn default() -> Self {
        Self {
            fps: DEFAULT_FPS,
            start_frame: DEFAULT_START_FRAME,
            cache_budget_bytes: DEFAULT_CACHE_BUDGET_MB as u64 * 1024 * 1024,
        }
    }
}

struct CacheEntry {
    snapshot: Snapshot,
    last_used: u64,
}

/// The resources one `goto` call borrows.
struct Env<'a> {
    graph: &'a Graph,
    gpu: &'a GpuContext,
    pool: &'a mut FieldPool,
    pipelines: &'a mut PipelineCache,
    dims: FieldDims,
}

/// Owns a graph's persistent state and produces any frame on request,
/// bit-identical to what forward playback would give.
///
/// A cached snapshot for frame N is the state *entering* N, taken before N's
/// `eval`. So producing N always means restoring or reaching the state entering
/// N, then running exactly one `eval`. A cache hit therefore costs one step,
/// and outputs never need caching.
pub struct Timeline {
    config: TimelineConfig,
    state: StateStore,
    /// The frame the held state is entering, or `None` if it is not usable.
    cursor: Option<u32>,
    cache: BTreeMap<u32, CacheEntry>,
    cache_bytes: u64,
    /// Never evicted: the snapshot most recently restored.
    protected: Option<u32>,
    clock: u64,
    caching: bool,
    steps: u64,
    warning: Option<String>,
}

impl Timeline {
    pub fn new(config: TimelineConfig) -> Self {
        Self {
            config,
            state: StateStore::new(),
            cursor: None,
            cache: BTreeMap::new(),
            cache_bytes: 0,
            protected: None,
            clock: 0,
            caching: true,
            steps: 0,
            warning: None,
        }
    }

    pub fn config(&self) -> TimelineConfig {
        self.config
    }

    /// Total `eval_frame` calls made so far.
    pub fn steps_run(&self) -> u64 {
        self.steps
    }

    /// Frames whose entering state is cached, in ascending order.
    pub fn cached_frames(&self) -> Vec<u32> {
        self.cache.keys().copied().collect()
    }

    pub fn caching_enabled(&self) -> bool {
        self.caching
    }

    /// GPU memory the snapshot cache currently holds.
    pub fn cached_bytes(&self) -> u64 {
        self.cache_bytes
    }

    /// A message the user should see, at most once.
    pub fn take_warning(&mut self) -> Option<String> {
        self.warning.take()
    }

    /// Forget all state and cached frames, returning their textures to `pool`.
    pub fn reset(&mut self, pool: &mut FieldPool) {
        self.state.clear(pool);
        for (_, entry) in std::mem::take(&mut self.cache) {
            entry.snapshot.release_to(pool);
        }
        self.cache_bytes = 0;
        self.cursor = None;
        self.protected = None;
        self.caching = true;
    }

    /// Forget all state and cached frames without pooling their textures.
    /// For use after the GPU device is lost, when they are worthless.
    pub fn discard(&mut self) {
        self.state = StateStore::new();
        self.cache.clear();
        self.cache_bytes = 0;
        self.cursor = None;
        self.protected = None;
    }

    /// Produce `frame`. Frames before `start_frame` produce `start_frame`.
    ///
    /// The caller owns the returned value and should release it to `pool`.
    pub fn goto(
        &mut self,
        graph: &Graph,
        gpu: &GpuContext,
        pool: &mut FieldPool,
        pipelines: &mut PipelineCache,
        dims: FieldDims,
        frame: u32,
    ) -> Result<Evaluated, NodeError> {
        let frame = frame.max(self.config.start_frame);

        if !graph.is_stateful() {
            let time = self.time(frame);
            self.steps += 1;
            return graph.eval_frame(gpu, pool, pipelines, &mut self.state, time, dims);
        }

        let mut env = Env {
            graph,
            gpu,
            pool,
            pipelines,
            dims,
        };
        match self.advance(&mut env, frame) {
            // Stored state no longer fits the domain: start over, once.
            Err(NodeError::StateShape { .. }) => {
                self.reset(env.pool);
                self.advance(&mut env, frame)
            }
            other => other,
        }
    }

    fn time(&self, frame: u32) -> Time {
        Time::at(frame, self.config.start_frame, self.config.fps)
    }

    fn advance(&mut self, env: &mut Env<'_>, frame: u32) -> Result<Evaluated, NodeError> {
        let result = self.advance_inner(env, frame);
        if result.is_err() {
            // A failed eval may have taken state out without putting it back.
            self.state.clear(env.pool);
            self.cursor = None;
        }
        result
    }

    fn advance_inner(&mut self, env: &mut Env<'_>, frame: u32) -> Result<Evaluated, NodeError> {
        if self.cursor != Some(frame) {
            let held = self.cursor.filter(|&c| c <= frame);
            let cached = if self.caching {
                self.cache.range(..=frame).next_back().map(|(&f, _)| f)
            } else {
                None
            };
            match (held, cached) {
                (Some(h), Some(c)) if h >= c => {}
                (_, Some(c)) => self.restore(env, c)?,
                (Some(_), None) => {}
                (None, None) => {
                    self.state.clear(env.pool);
                    self.cursor = Some(self.config.start_frame);
                }
            }
            while let Some(c) = self.cursor.filter(|&c| c < frame) {
                let stepped = self.step(env, c)?;
                stepped.value.release_to(env.pool);
            }
        }
        self.step(env, frame)
    }

    fn restore(&mut self, env: &mut Env<'_>, frame: u32) -> Result<(), NodeError> {
        self.clock += 1;
        let Some(entry) = self.cache.get_mut(&frame) else {
            // Unreachable: callers pass a key they just found. Fall back to a
            // full re-simulation rather than trusting the held state.
            self.state.clear(env.pool);
            self.cursor = Some(self.config.start_frame);
            return Ok(());
        };
        entry.last_used = self.clock;
        self.state.restore(&entry.snapshot, env.gpu, env.pool)?;
        self.cursor = Some(frame);
        self.protected = Some(frame);
        Ok(())
    }

    /// Evaluate `frame` from the state entering it, caching that state first.
    fn step(&mut self, env: &mut Env<'_>, frame: u32) -> Result<Evaluated, NodeError> {
        if self.caching && !self.cache.contains_key(&frame) {
            let snapshot = self.state.snapshot(env.gpu, env.pool)?;
            self.insert(frame, snapshot, env.pool);
        }
        let time = self.time(frame);
        let out = env.graph.eval_frame(
            env.gpu,
            env.pool,
            env.pipelines,
            &mut self.state,
            time,
            env.dims,
        )?;
        self.steps += 1;
        // `None` past u32::MAX: the held state then matches no frame.
        self.cursor = frame.checked_add(1);
        Ok(out)
    }

    fn insert(&mut self, frame: u32, snapshot: Snapshot, pool: &mut FieldPool) {
        let budget = self.config.cache_budget_bytes;
        let bytes = snapshot.bytes();

        if bytes > budget {
            self.caching = false;
            self.warning = Some(format!(
                "one frame of simulation state is {bytes} bytes, more than the whole \
                 {budget}-byte cache budget; frame caching is off, so scrubbing \
                 backwards re-simulates from the start frame"
            ));
            snapshot.release_to(pool);
            for (_, entry) in std::mem::take(&mut self.cache) {
                entry.snapshot.release_to(pool);
            }
            self.cache_bytes = 0;
            return;
        }

        while self.cache_bytes + bytes > budget {
            let victim = self
                .cache
                .iter()
                .filter(|(f, _)| Some(**f) != self.protected)
                .min_by_key(|(_, e)| e.last_used)
                .map(|(f, _)| *f);
            let Some(victim) = victim else {
                // Only the protected snapshot is left, and the new one does not fit beside it.
                snapshot.release_to(pool);
                return;
            };
            if let Some(entry) = self.cache.remove(&victim) {
                self.cache_bytes -= entry.snapshot.bytes();
                entry.snapshot.release_to(pool);
            }
        }

        self.clock += 1;
        self.cache_bytes += bytes;
        self.cache.insert(
            frame,
            CacheEntry {
                snapshot,
                last_used: self.clock,
            },
        );
    }
}
