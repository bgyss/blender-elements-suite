use elements_core::gpu::{FieldDims, FieldPool, GpuContext, PipelineCache};
use elements_core::graph::{Document, Graph, NodeRegistry, Timeline, TimelineConfig};

const ACCUMULATE_DOC: &str = r#"{
  "version": 1,
  "dims": [4, 4, 4],
  "nodes": [
    { "id": 0, "kind": "core.constant_field", "params": { "value": 0.5 } },
    { "id": 1, "kind": "core.accumulate", "params": {} },
    { "id": 2, "kind": "core.output", "params": {} }
  ],
  "edges": [
    { "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 },
    { "from_node": 1, "from_index": 0, "to_node": 2, "to_index": 0 }
  ],
  "output": 2
}"#;

const CONSTANT_DOC: &str = r#"{
  "version": 1,
  "dims": [4, 4, 4],
  "nodes": [
    { "id": 0, "kind": "core.constant_field", "params": { "value": 0.5 } },
    { "id": 1, "kind": "core.output", "params": {} }
  ],
  "edges": [{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }],
  "output": 1
}"#;

/// One 4³ R32Float field.
const SNAPSHOT_BYTES: u64 = 4 * 4 * 4 * 4;

struct Harness {
    gpu: GpuContext,
    pool: FieldPool,
    pipelines: PipelineCache,
    graph: Graph,
    dims: FieldDims,
}

impl Harness {
    fn new(doc: &str) -> Self {
        let (graph, dims) = Document::from_json(doc)
            .unwrap()
            .into_graph(&NodeRegistry::with_builtins())
            .unwrap();
        Self {
            gpu: GpuContext::new_headless().expect("no GPU adapter available"),
            pool: FieldPool::new(),
            pipelines: PipelineCache::new(),
            graph,
            dims,
        }
    }

    fn goto_at(&mut self, timeline: &mut Timeline, frame: u32, dims: FieldDims) -> Vec<f32> {
        let evaluated = timeline
            .goto(
                &self.graph,
                &self.gpu,
                &mut self.pool,
                &mut self.pipelines,
                dims,
                frame,
            )
            .unwrap();
        let values = evaluated
            .value
            .as_field()
            .unwrap()
            .read_back(&self.gpu)
            .unwrap();
        evaluated.value.release_to(&mut self.pool);
        values
    }

    fn goto(&mut self, timeline: &mut Timeline, frame: u32) -> Vec<f32> {
        let dims = self.dims;
        self.goto_at(timeline, frame, dims)
    }
}

fn timeline(cache_budget_bytes: u64) -> Timeline {
    Timeline::new(TimelineConfig {
        fps: 24.0,
        start_frame: 1,
        cache_budget_bytes,
    })
}

fn expected(frame: u32) -> f32 {
    0.5 * frame as f32 / 24.0
}

fn assert_frame(values: &[f32], frame: u32) {
    let want = expected(frame);
    for &v in values {
        assert!(
            (v - want).abs() < 1e-6,
            "frame {frame}: expected {want}, got {v}"
        );
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn playing_forward_matches_the_closed_form() {
    let mut h = Harness::new(ACCUMULATE_DOC);
    let mut tl = timeline(1 << 30);
    for frame in 1..=10 {
        let values = h.goto(&mut tl, frame);
        assert_frame(&values, frame);
    }
    assert_eq!(
        tl.steps_run(),
        10,
        "forward playback costs one step per frame"
    );
}

#[test]
fn scrubbing_back_to_a_cached_frame_costs_one_step() {
    let mut h = Harness::new(ACCUMULATE_DOC);
    let mut tl = timeline(1 << 30);
    h.goto(&mut tl, 10);
    let before = tl.steps_run();
    let values = h.goto(&mut tl, 3);
    assert_frame(&values, 3);
    assert_eq!(tl.steps_run(), before + 1);
}

#[test]
fn resuming_from_an_earlier_cached_frame_steps_forward_correctly() {
    let mut h = Harness::new(ACCUMULATE_DOC);
    let mut tl = timeline(1 << 30);
    h.goto(&mut tl, 5);
    h.goto(&mut tl, 3);
    // The held state is entering 4. Frame 5's entering state is cached and is
    // later, so the timeline restores it and steps through 5 and 6 to 7.
    let values = h.goto(&mut tl, 7);
    assert_frame(&values, 7);
    assert_eq!(tl.steps_run(), 5 + 1 + 3);
}

#[test]
fn the_cache_stays_within_budget_and_evicted_frames_re_simulate() {
    let mut h = Harness::new(ACCUMULATE_DOC);
    // Room for three 4³ snapshots (the start frame's empty snapshot is free).
    let budget = 3 * SNAPSHOT_BYTES;
    let mut tl = timeline(budget);
    h.goto(&mut tl, 10);
    assert!(
        tl.cached_bytes() <= budget,
        "{} bytes cached against a {budget}-byte budget",
        tl.cached_bytes()
    );
    assert!(
        tl.cached_frames().contains(&10),
        "the newest frame survives eviction"
    );
    assert!(
        !tl.cached_frames().contains(&5),
        "frame 5 must have been evicted for this test to mean anything: {:?}",
        tl.cached_frames()
    );
    let values = h.goto(&mut tl, 5);
    assert_frame(&values, 5);
}

#[test]
fn a_frame_is_bit_identical_however_it_is_reached() {
    let mut h = Harness::new(ACCUMULATE_DOC);

    let mut direct = timeline(1 << 30);
    let want = bits(&h.goto(&mut direct, 10));

    let mut wandering = timeline(1 << 30);
    h.goto(&mut wandering, 15);
    h.goto(&mut wandering, 3);
    let got = bits(&h.goto(&mut wandering, 10));

    assert_eq!(got, want);
}

#[test]
fn frames_before_the_start_are_the_start_frame() {
    let mut h = Harness::new(ACCUMULATE_DOC);
    let mut tl = timeline(1 << 30);
    assert_frame(&h.goto(&mut tl, 0), 1);
    assert_frame(&h.goto(&mut tl, 1), 1);
}

#[test]
fn a_budget_smaller_than_one_snapshot_disables_caching_but_stays_correct() {
    let mut h = Harness::new(ACCUMULATE_DOC);
    let mut tl = timeline(SNAPSHOT_BYTES - 1);
    assert_frame(&h.goto(&mut tl, 5), 5);
    assert!(!tl.caching_enabled());
    assert!(
        tl.take_warning().is_some(),
        "the user must be told scrubbing is slow"
    );
    assert!(tl.cached_frames().is_empty());
    assert_frame(&h.goto(&mut tl, 2), 2);
}

#[test]
fn a_domain_change_resets_the_simulation() {
    let mut h = Harness::new(ACCUMULATE_DOC);
    let mut tl = timeline(1 << 30);
    h.goto(&mut tl, 3);
    // Same graph, bigger domain: every cached snapshot is now the wrong shape.
    let values = h.goto_at(&mut tl, 3, FieldDims::new(8, 8, 8));
    assert_eq!(values.len(), 8 * 8 * 8);
    assert_frame(&values, 3);
}

#[test]
fn a_stateless_graph_is_evaluated_directly() {
    let mut h = Harness::new(CONSTANT_DOC);
    let mut tl = timeline(1 << 30);
    for frame in [7, 2, 9] {
        let values = h.goto(&mut tl, frame);
        assert!(values.iter().all(|&v| v == 0.5));
    }
    assert!(tl.cached_frames().is_empty());
    assert_eq!(tl.steps_run(), 3);
}
