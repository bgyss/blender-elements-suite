//! FT4 surface ignition (spec 2026-09-30-flamethrower-ft4-surface-ignition-design.md).

mod common;

use common::*;
use elements_core::gpu::{FieldDims, FieldPool, PipelineCache};
use elements_ember::collider::{ColliderFields, ColliderParams, SurfaceFuel, fill_collider};
use elements_ember::transform::{Key, Shape, Transform};

const DX: f32 = 0.125;

fn still_box(surface_fuel: Option<SurfaceFuel>) -> ColliderParams {
    ColliderParams {
        shape: Shape::Box {
            half_extents: [0.25, 0.25, 0.125],
        },
        transform: Transform {
            keys: vec![Key {
                frame: 0.0,
                translate: [0.5, 0.5, 0.5],
                rotate: None,
            }],
        },
        surface_fuel,
    }
}

#[test]
fn a_collider_outputs_its_load_where_the_sdf_is_negative() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(8, 8, 8);
    let sdf = pool
        .acquire(&gpu, cells, elements_core::gpu::FieldFormat::R32Float)
        .unwrap();
    let load = pool
        .acquire(&gpu, cells, elements_core::gpu::FieldFormat::R32Float)
        .unwrap();
    let velocity = pool.acquire_staggered_uninit(&gpu, cells).unwrap();
    let params = still_box(Some(SurfaceFuel { load: 7.5 }));
    let pose = params.transform.pose(0.0, 1.0 / 24.0);
    fill_collider(
        &gpu,
        &mut cache,
        &params,
        &pose,
        DX,
        ColliderFields {
            sdf: &sdf,
            velocity: &velocity,
        },
    )
    .unwrap();
    elements_ember::collider::fill_surface_load(&gpu, &mut cache, &sdf, 7.5, &load).unwrap();
    let sdf_data = sdf.read_back(&gpu).unwrap();
    let load_data = load.read_back(&gpu).unwrap();
    let inside = sdf_data.iter().filter(|&&d| d < 0.0).count();
    assert!(inside > 0, "the box must cover some cells");
    for (d, l) in sdf_data.iter().zip(&load_data) {
        let want = if *d < 0.0 { 7.5 } else { 0.0 };
        assert_eq!(*l, want, "sdf {d}");
    }
}

#[test]
fn surface_fuel_rejects_a_moving_collider_and_a_negative_load() {
    let moving = serde_json::json!({
        "shape": { "sphere": { "radius": 0.1 } },
        "transform": { "keys": [
            { "frame": 0, "translate": [0.0, 0.0, 0.0] },
            { "frame": 10, "translate": [1.0, 0.0, 0.0] } ] },
        "surface_fuel": { "load": 3.0 }
    });
    assert!(
        elements_ember::registry()
            .build("ember.collider", &moving)
            .is_err()
    );
    let negative = serde_json::json!({
        "shape": { "sphere": { "radius": 0.1 } },
        "transform": { "keys": [{ "frame": 0, "translate": [0.0, 0.0, 0.0] }] },
        "surface_fuel": { "load": -1.0 }
    });
    assert!(
        elements_ember::registry()
            .build("ember.collider", &negative)
            .is_err()
    );
    let still = serde_json::json!({
        "shape": { "sphere": { "radius": 0.1 } },
        "transform": { "keys": [{ "frame": 0, "translate": [0.0, 0.0, 0.0] }] },
        "surface_fuel": { "load": 3.0 }
    });
    assert!(
        elements_ember::registry()
            .build("ember.collider", &still)
            .is_ok()
    );
}

#[test]
fn a_union_keeps_the_larger_load_and_a_missing_side_counts_as_zero() {
    use elements_ember::unions::union_surface_load;
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(4, 4, 4);
    let n = cells.voxel_count();
    let a: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| ((i * 3) % 4) as f32).collect();
    let fa = upload(&gpu, &mut pool, cells, &a);
    let fb = upload(&gpu, &mut pool, cells, &b);
    let out = upload(&gpu, &mut pool, cells, &vec![-1.0; n]);
    union_surface_load(&gpu, &mut cache, &fa, &fb, &out).unwrap();
    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x.max(*y)).collect();
    assert_eq!(out.read_back(&gpu).unwrap(), want);
}

#[test]
fn a_union_with_one_unconnected_load_outputs_the_other() {
    let doc = r#"{ "version": 3, "dims": [8, 8, 8], "fps": 24.0, "domain_size": 1.0,
      "nodes": [
        { "id": 0, "kind": "ember.collider", "params": {
            "shape": { "box": { "half_extents": [0.2, 0.2, 0.1] } },
            "transform": { "keys": [{ "frame": 0, "translate": [0.5, 0.5, 0.5] }] },
            "surface_fuel": { "load": 4.0 } } },
        { "id": 1, "kind": "ember.collider", "params": {
            "shape": { "sphere": { "radius": 0.05 } },
            "transform": { "keys": [{ "frame": 0, "translate": [0.1, 0.1, 0.1] }] } } },
        { "id": 2, "kind": "ember.collider_union", "params": {} },
        { "id": 3, "kind": "core.output", "params": {} } ],
      "edges": [
        { "from_node": 0, "from_index": 0, "to_node": 2, "to_index": 0 },
        { "from_node": 0, "from_index": 1, "to_node": 2, "to_index": 1 },
        { "from_node": 1, "from_index": 0, "to_node": 2, "to_index": 2 },
        { "from_node": 1, "from_index": 1, "to_node": 2, "to_index": 3 },
        { "from_node": 0, "from_index": 2, "to_node": 2, "to_index": 4 },
        { "from_node": 2, "from_index": 2, "to_node": 3, "to_index": 0 } ],
      "output": 3 }"#;
    let mut s = Session::new(doc);
    let mut t = timeline(0);
    let bits = s.density_bits(&mut t, 1);
    let loads: Vec<f32> = bits.iter().map(|b| f32::from_bits(*b)).collect();
    assert!(loads.contains(&4.0), "the first collider's load survives");
    assert!(loads.iter().all(|&l| l == 0.0 || l == 4.0), "{loads:?}");
}

/// FT4 spec §3.5: with the surface unconnected, a fire document is bit-identical
/// to the one before FT4. Recorded on the commit before any FT4 solver change,
/// on this adapter; other adapters print and skip.
#[test]
fn without_a_surface_a_fire_document_matches_the_solver_before_ft4() {
    const RECORDED_ON: &str = "Apple M1 Max";
    const WANT: u64 = 0xfbcd_e02f_2e02_d31f;
    let ctx = gpu();
    if ctx.adapter_name() != RECORDED_ON {
        eprintln!(
            "skipped: recorded on {RECORDED_ON}, this is {}",
            ctx.adapter_name()
        );
        return;
    }
    let mut s = Session::new(&fire_doc());
    let mut t = timeline(0);
    let mut bits = Vec::new();
    for frame in 1..=40 {
        bits = s.density_bits(&mut t, frame);
    }
    assert_eq!(fnv1a(&bits), WANT, "frame 40 density changed");
}

fn fnv1a(bits: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bits {
        for byte in b.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// A 16³ plume fed fuel by a second emitter, density (socket 0) as the output.
fn fire_doc() -> String {
    r#"{ "version": 3, "dims": [16, 16, 16], "fps": 24.0, "domain_size": 2.0,
      "nodes": [
        { "id": 0, "kind": "ember.sphere_emitter", "params": { "center": [1.0, 1.0, 0.4],
          "radius": 0.3, "density_rate": 1.0, "temperature_rate": 2.0 } },
        { "id": 1, "kind": "ember.smoke_solver", "params": { "buoyancy_temperature": 1.0 } },
        { "id": 2, "kind": "core.output", "params": {} },
        { "id": 3, "kind": "ember.sphere_emitter", "params": { "center": [1.0, 1.0, 0.4],
          "radius": 0.3, "density_rate": 1.0, "temperature_rate": 0.0 } } ],
      "edges": [
        { "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 },
        { "from_node": 0, "from_index": 1, "to_node": 1, "to_index": 1 },
        { "from_node": 1, "from_index": 0, "to_node": 2, "to_index": 0 },
        { "from_node": 3, "from_index": 0, "to_node": 1, "to_index": 6 } ],
      "output": 2 }"#
        .to_owned()
}

#[test]
fn surface_burn_rate_must_be_finite_and_non_negative() {
    use elements_ember::solver::resolve_params;
    assert_eq!(
        resolve_params(&serde_json::json!({}))
            .unwrap()
            .surface_burn_rate,
        2.0
    );
    assert!(resolve_params(&serde_json::json!({ "surface_burn_rate": 0.5 })).is_ok());
    assert!(resolve_params(&serde_json::json!({ "surface_burn_rate": -1.0 })).is_err());
    // JSON cannot carry NaN (`json!` turns it into null, which means "default"),
    // so a non-finite rate arrives as a number that overflows f32 to infinity.
    assert!(resolve_params(&serde_json::json!({ "surface_burn_rate": 1e39 })).is_err());
}

// ---- Surface kernels (FT4 spec §3.3) ----

use elements_core::gpu::{ComputeBatch, Field, FieldFormat, GpuContext};
use elements_ember::kernels::{
    Solids, StepConstants, Uniforms, surface_burn, surface_char, surface_gather,
};

const K: FieldDims = FieldDims { x: 8, y: 8, z: 8 };
const H: f32 = 1.0 / 48.0;
const IGN: f32 = 1.5;
const BURN_RATE: f32 = 2.0;

/// A vertical plank: solid cells i = 4, j in 2..6, k in 1..7, each with `load`.
fn plank(load: f32) -> (Vec<f32>, Vec<f32>) {
    let mut mask = vec![0.0; K.voxel_count()];
    let mut loads = vec![0.0; K.voxel_count()];
    for k in 1..7 {
        for j in 2..6 {
            mask[index(K, 4, j, k)] = 1.0;
            loads[index(K, 4, j, k)] = load;
        }
    }
    (mask, loads)
}

fn constants() -> StepConstants {
    StepConstants {
        has_solids: true,
        surface_burn_rate: BURN_RATE,
        ignition_temperature: IGN,
        ..StepConstants::new(K, H, 0.125)
    }
}

fn in_domain(c: [i32; 3]) -> bool {
    c.iter().all(|&v| (0..8).contains(&v))
}

fn neighbours(c: [i32; 3]) -> [[i32; 3]; 6] {
    let mut out = [c; 6];
    for n in 0..6 {
        out[n][n / 2] += if n % 2 == 1 { 1 } else { -1 };
    }
    out
}

fn fluid(mask: &[f32], c: [i32; 3]) -> bool {
    in_domain(c) && mask[index(K, c[0] as u32, c[1] as u32, c[2] as u32)] <= 0.5
}

fn at(c: [i32; 3]) -> usize {
    index(K, c[0] as u32, c[1] as u32, c[2] as u32)
}

/// FT4 spec §3.3, directly: returns (burned', emitted, rate).
fn cpu_surface(
    mask: &[f32],
    load: &[f32],
    burned: &[f32],
    temperature: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = K.voxel_count();
    let (mut b2, mut emitted, mut rate) = (burned.to_vec(), vec![0.0; n], vec![0.0; n]);
    let burn = BURN_RATE * H;
    for k in 0..8 {
        for j in 0..8 {
            for i in 0..8 {
                let c = [i, j, k];
                let fluid_n: Vec<[i32; 3]> = neighbours(c)
                    .into_iter()
                    .filter(|&q| fluid(mask, q))
                    .collect();
                if mask[at(c)] > 0.5 && load[at(c)] > 0.0 && !fluid_n.is_empty() {
                    let hottest = fluid_n
                        .iter()
                        .map(|&q| temperature[at(q)])
                        .fold(f32::MIN, f32::max);
                    let b = burned[at(c)];
                    if (b > 0.0 || hottest > IGN) && b < load[at(c)] {
                        let e = burn.min(load[at(c)] - b);
                        b2[at(c)] = b + e;
                        emitted[at(c)] = e;
                    }
                }
            }
        }
    }
    for k in 0..8 {
        for j in 0..8 {
            for i in 0..8 {
                let c = [i, j, k];
                if fluid(mask, c) {
                    let mut sum = 0.0;
                    for q in neighbours(c) {
                        if in_domain(q) && mask[at(q)] > 0.5 && emitted[at(q)] > 0.0 {
                            let nf = neighbours(q)
                                .into_iter()
                                .filter(|&r| fluid(mask, r))
                                .count();
                            sum += emitted[at(q)] / nf as f32;
                        }
                    }
                    rate[at(c)] = sum / H;
                }
            }
        }
    }
    (b2, emitted, rate)
}

struct Kernels {
    gpu: GpuContext,
    pool: FieldPool,
    cache: PipelineCache,
    mask: Field,
    velocity: elements_core::gpu::StaggeredField,
    load: Field,
    burned: Field,
    emitted: Field,
    rate: Field,
}

impl Kernels {
    fn new(mask: &[f32], load: &[f32], burned: &[f32]) -> Self {
        let gpu = gpu();
        let mut pool = FieldPool::new();
        let mut cache = PipelineCache::new();
        let velocity = pool.acquire_staggered_zeroed(&gpu, &mut cache, K).unwrap();
        let mask = upload(&gpu, &mut pool, K, mask);
        let load = upload(&gpu, &mut pool, K, load);
        let burned = upload(&gpu, &mut pool, K, burned);
        let emitted = upload(&gpu, &mut pool, K, &vec![9.0; K.voxel_count()]);
        let rate = upload(&gpu, &mut pool, K, &vec![9.0; K.voxel_count()]);
        Self {
            gpu,
            pool,
            cache,
            mask,
            velocity,
            load,
            burned,
            emitted,
            rate,
        }
    }

    /// One substep of the two kernels; returns (burned', emitted, rate).
    fn run(&mut self, temperature: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let temp = upload(&self.gpu, &mut self.pool, K, temperature);
        let u = Uniforms::new(&self.gpu, &constants()).unwrap();
        let solids = Solids {
            mask: &self.mask,
            velocity: &self.velocity,
        };
        let mut batch = ComputeBatch::new();
        surface_burn(
            &self.gpu,
            &mut self.cache,
            &mut batch,
            &u,
            &self.load,
            &temp,
            solids,
            &self.burned,
            &self.emitted,
        )
        .unwrap();
        surface_gather(
            &self.gpu,
            &mut self.cache,
            &mut batch,
            &u,
            solids,
            &self.emitted,
            &self.rate,
        )
        .unwrap();
        batch.submit(&self.gpu).unwrap();
        (
            self.burned.read_back(&self.gpu).unwrap(),
            self.emitted.read_back(&self.gpu).unwrap(),
            self.rate.read_back(&self.gpu).unwrap(),
        )
    }
}

fn gas(temperature_at: &[([i32; 3], f32)]) -> Vec<f32> {
    let mut t = vec![0.0; K.voxel_count()];
    for (c, v) in temperature_at {
        t[at(*c)] = *v;
    }
    t
}

#[test]
fn cold_gas_ignites_nothing() {
    let (mask, load) = plank(5.0);
    let mut k = Kernels::new(&mask, &load, &vec![0.0; K.voxel_count()]);
    // Even gas exactly at the threshold does not ignite: strictly above.
    let t = vec![IGN; K.voxel_count()];
    for _ in 0..10 {
        let (burned, emitted, rate) = k.run(&t);
        assert!(burned.iter().all(|&b| b == 0.0));
        assert!(emitted.iter().all(|&e| e == 0.0));
        assert!(rate.iter().all(|&r| r == 0.0));
    }
}

#[test]
fn gas_above_the_threshold_ignites_only_its_neighbour() {
    let (mask, load) = plank(5.0);
    let mut k = Kernels::new(&mask, &load, &vec![0.0; K.voxel_count()]);
    // One hot fluid cell beside plank cell (4, 3, 3).
    let t = gas(&[([3, 3, 3], IGN + 0.01)]);
    let (burned, emitted, rate) = k.run(&t);
    let burn = BURN_RATE * H;
    assert_eq!(burned[at([4, 3, 3])], burn);
    assert_eq!(burned.iter().filter(|&&b| b > 0.0).count(), 1);
    assert_eq!(emitted[at([4, 3, 3])], burn);
    // Its fluid neighbours: (3,3,3), (5,3,3), (4,2,3)? no: j = 2 is plank.
    // The kernel and the CPU reference must agree on the split.
    let (cb, ce, cr) = cpu_surface(&mask, &load, &vec![0.0; K.voxel_count()], &t);
    assert_close(&burned, &cb, 1e-7, "burned");
    assert_close(&emitted, &ce, 1e-7, "emitted");
    assert_close(&rate, &cr, 1e-4, "rate");
}

#[test]
fn the_kernels_match_the_cpu_reference_over_many_substeps() {
    let (mask, load) = plank(0.3); // 7.2 substeps of burning: spent late in the cooling phase
    let mut k = Kernels::new(&mask, &load, &vec![0.0; K.voxel_count()]);
    let mut burned = vec![0.0; K.voxel_count()];
    let hot = gas(&[([3, 3, 3], 2.0), ([5, 4, 5], 2.0), ([3, 5, 1], 1.6)]);
    let cold = vec![0.0; K.voxel_count()];
    // 6 substeps of hot gas, then 10 of cold: lit cells keep burning to depletion.
    for step in 0..16 {
        let gas = if step < 6 { &hot } else { &cold };
        let (want_b, want_e, want_r) = cpu_surface(&mask, &load, &burned, gas);
        let (got_b, got_e, got_r) = k.run(gas);
        assert_close(&got_b, &want_b, 1e-7, &format!("burned step {step}"));
        assert_close(&got_e, &want_e, 1e-7, &format!("emitted step {step}"));
        assert_close(&got_r, &want_r, 1e-4, &format!("rate step {step}"));
        burned = want_b;
    }
}

#[test]
fn the_reservoir_depletes_and_emission_stops() {
    let (mask, load) = plank(0.1);
    let mut k = Kernels::new(&mask, &load, &vec![0.0; K.voxel_count()]);
    let hot = gas(&[([3, 3, 3], 2.0)]);
    let mut last = 0.0f32;
    for _ in 0..30 {
        let (burned, emitted, _) = k.run(&hot);
        assert!(burned[at([4, 3, 3])] <= 0.1 + 1e-7, "never past the load");
        last = emitted[at([4, 3, 3])];
    }
    let (burned, _, _) = k.run(&hot);
    assert_eq!(burned[at([4, 3, 3])], 0.1, "fully burned");
    assert_eq!(last, 0.0, "a spent cell emits nothing");
}

#[test]
fn what_the_wood_loses_the_gas_gains() {
    let (mask, load) = plank(3.0);
    let mut k = Kernels::new(&mask, &load, &vec![0.0; K.voxel_count()]);
    let hot = gas(&[([3, 3, 3], 2.0), ([5, 4, 5], 2.0)]);
    let (burned, _, rate) = k.run(&hot);
    let lost: f64 = burned.iter().map(|&b| f64::from(b)).sum();
    let gained: f64 = rate.iter().map(|&r| f64::from(r) * f64::from(H)).sum();
    assert!(lost > 0.0);
    assert!(
        (lost - gained).abs() <= 1e-6 * lost,
        "lost {lost}, gained {gained}"
    );
}

#[test]
fn char_is_the_burned_fraction_and_zero_without_a_load() {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let n = K.voxel_count();
    let mut burned = vec![0.0; n];
    let mut load = vec![0.0; n];
    burned[0] = 1.0;
    load[0] = 4.0; // quarter burned
    burned[1] = 4.0;
    load[1] = 4.0; // spent
    burned[2] = 0.5; // burned but no load: must be 0, not NaN or inf
    let b = upload(&gpu, &mut pool, K, &burned);
    let l = upload(&gpu, &mut pool, K, &load);
    let dst = pool.acquire(&gpu, K, FieldFormat::R32Float).unwrap();
    let u = Uniforms::new(&gpu, &constants()).unwrap();
    let mut batch = ComputeBatch::new();
    surface_char(&gpu, &mut cache, &mut batch, &u, &b, &l, &dst).unwrap();
    batch.submit(&gpu).unwrap();
    let out = dst.read_back(&gpu).unwrap();
    assert_eq!(&out[..3], &[0.25, 1.0, 0.0]);
    assert!(out.iter().all(|v| v.is_finite()));
}

// ---- Solver integration (FT4 spec §3.1, §3.2, §4) ----

/// A 16³ domain with a static wooden slab (surface_fuel) at x ≈ 1.0, a
/// temperature emitter touching it, and the fuel input connected at rate 0.
/// `socket` of the solver goes to the output. Options cut one connection.
fn slab_doc(socket: u32, heat: f32, connect_fuel: bool, connect_load: bool) -> String {
    let fuel_edge = if connect_fuel {
        r#",{ "from_node": 3, "from_index": 0, "to_node": 1, "to_index": 6 }"#
    } else {
        ""
    };
    let load_edge = if connect_load {
        r#",{ "from_node": 4, "from_index": 2, "to_node": 1, "to_index": 7 }"#
    } else {
        ""
    };
    format!(
        r#"{{ "version": 3, "dims": [16, 16, 16], "fps": 24.0, "domain_size": 2.0,
      "nodes": [
        {{ "id": 0, "kind": "ember.sphere_emitter", "params": {{ "center": [0.75, 1.0, 0.6],
           "radius": 0.2, "density_rate": 0.0, "temperature_rate": {heat:?} }} }},
        {{ "id": 1, "kind": "ember.smoke_solver", "params": {{ "buoyancy_temperature": 1.0 }} }},
        {{ "id": 2, "kind": "core.output", "params": {{}} }},
        {{ "id": 3, "kind": "ember.sphere_emitter", "params": {{ "center": [1.0, 1.0, 1.4],
           "radius": 0.1, "density_rate": 0.0, "temperature_rate": 0.0 }} }},
        {{ "id": 4, "kind": "ember.collider", "params": {{
           "shape": {{ "box": {{ "half_extents": [0.07, 0.4, 0.4] }} }},
           "transform": {{ "keys": [{{ "frame": 0, "translate": [1.0, 1.0, 0.6] }}] }},
           "surface_fuel": {{ "load": 4.0 }} }} }} ],
      "edges": [
        {{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }},
        {{ "from_node": 0, "from_index": 1, "to_node": 1, "to_index": 1 }},
        {{ "from_node": 4, "from_index": 0, "to_node": 1, "to_index": 4 }},
        {{ "from_node": 4, "from_index": 1, "to_node": 1, "to_index": 5 }},
        {{ "from_node": 1, "from_index": {socket}, "to_node": 2, "to_index": 0 }}{fuel_edge}{load_edge} ],
      "output": 2 }}"#
    )
}

fn char_at_frame(doc: &str, frame: u32) -> Vec<f32> {
    let mut s = Session::new(doc);
    let mut t = timeline(0);
    s.density_bits(&mut t, frame)
        .iter()
        .map(|b| f32::from_bits(*b))
        .collect()
}

#[test]
fn a_slab_with_no_heat_never_chars() {
    let char = char_at_frame(&slab_doc(4, 0.0, true, true), 20);
    assert!(char.iter().all(|&c| c == 0.0), "no heat, no ignition");
}

#[test]
fn a_slab_beside_a_heat_source_chars_and_stays_in_range() {
    let char = char_at_frame(&slab_doc(4, 30.0, true, true), 20);
    let charred = char.iter().filter(|&&c| c > 0.0).count();
    let max = char.iter().copied().fold(0.0f32, f32::max);
    assert!(charred > 0, "a hot source must ignite the slab");
    assert!(
        char.iter().all(|&c| (0.0..=1.0).contains(&c)),
        "char in [0, 1]: {charred} charred cells, max {max}"
    );
}

#[test]
fn char_is_bit_identical_however_frame_40_is_reached() {
    // The helper also requires a nonzero output at frame 40: here, some char.
    assert_doc_frame_40_is_bit_identical(&slab_doc(4, 30.0, true, true));
}

#[test]
fn the_load_input_needs_fuel_and_a_collider() {
    use elements_core::graph::NodeError;
    fn eval_error(doc: &str) -> NodeError {
        let mut s = Session::new(doc);
        let mut t = timeline(0);
        match t.goto(&s.graph, &s.gpu, &mut s.pool, &mut s.pipelines, s.dims, 1) {
            Ok(_) => panic!("must fail"),
            Err(e) => e,
        }
    }
    // Load without fuel.
    let e = eval_error(&slab_doc(4, 1.0, false, true));
    assert!(
        matches!(
            e,
            NodeError::IncompletePair {
                connected: 7,
                missing: 6,
                ..
            }
        ),
        "{e:?}"
    );
    // Load without a collider: drop the collider edges.
    let doc = slab_doc(4, 1.0, true, true)
        .replace(
            r#"{ "from_node": 4, "from_index": 0, "to_node": 1, "to_index": 4 },"#,
            "",
        )
        .replace(
            r#"{ "from_node": 4, "from_index": 1, "to_node": 1, "to_index": 5 },"#,
            "",
        );
    let e = eval_error(&doc);
    assert!(
        matches!(
            e,
            NodeError::IncompletePair {
                connected: 7,
                missing: 4,
                ..
            }
        ),
        "{e:?}"
    );
}

/// A graph evaluated frame by frame against one state store, so a test can
/// swap the graph between frames (the timeline's `StateShape` reset path).
struct Stepper {
    gpu: GpuContext,
    pool: FieldPool,
    pipelines: PipelineCache,
    state: elements_core::graph::StateStore,
    graph: elements_core::graph::Graph,
    dims: FieldDims,
}

impl Stepper {
    fn new(doc: &str) -> Self {
        let s = Session::new(doc);
        Self {
            gpu: s.gpu,
            pool: s.pool,
            pipelines: s.pipelines,
            state: elements_core::graph::StateStore::new(),
            graph: s.graph,
            dims: s.dims,
        }
    }

    fn frame(&mut self, frame: u32) -> Result<(), elements_core::graph::NodeError> {
        let out = self.graph.eval_frame(
            &self.gpu,
            &mut self.pool,
            &mut self.pipelines,
            &mut self.state,
            elements_core::graph::Time::at(frame, 1, 24.0),
            self.dims,
        )?;
        out.value.release_to(&mut self.pool);
        Ok(())
    }
}

#[test]
fn connecting_the_load_changes_the_state_shape() {
    use elements_core::graph::{Document, NodeError};
    use elements_ember::solver::SURFACE;
    for (first, then) in [(true, false), (false, true)] {
        let mut a = Stepper::new(&slab_doc(4, 30.0, true, first));
        a.frame(1).unwrap();
        let (graph, _) = Document::from_json(&slab_doc(4, 30.0, true, then))
            .unwrap()
            .into_graph(&elements_ember::registry())
            .unwrap();
        a.graph = graph;
        match a.frame(2) {
            Err(NodeError::StateShape { slot, .. }) => assert_eq!(slot, SURFACE),
            other => panic!("{first:?} → {then:?}: {other:?}"),
        }
        // The mismatch path must return every field it took from the store.
        assert_eq!(
            a.pool.pooled_count() as u64,
            a.pool.allocation_count(),
            "{first:?} → {then:?}: the failed frame must return every taken field"
        );
    }
}

#[test]
fn a_full_substep_moves_the_wood_into_the_gas_fuel() {
    use elements_ember::solver::{PressureSolve, SolverState, Sources, substep};
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let (mask, load) = plank(3.0);
    let zero = vec![0.0; K.voxel_count()];
    let density_src = upload(&gpu, &mut pool, K, &zero);
    // Emitted at rate 2/h for one substep, the hot cell reaches 2 > IGN
    // before the burn reads it: substep emits temperature first.
    let heat = upload(&gpu, &mut pool, K, &gas(&[([3, 3, 3], 2.0 / H)]));
    let no_fuel = upload(&gpu, &mut pool, K, &zero);
    let load_f = upload(&gpu, &mut pool, K, &load);
    let mask_f = upload(&gpu, &mut pool, K, &mask);
    let velocity = pool.acquire_staggered_zeroed(&gpu, &mut cache, K).unwrap();
    let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, K).unwrap();
    state.add_fire(&gpu, &mut cache, &mut pool).unwrap();
    state.add_surface(&gpu, &mut cache, &mut pool).unwrap();
    let c = StepConstants {
        open_mask: 0,
        fire: true,
        has_solids: true,
        surface_burn_rate: BURN_RATE,
        ignition_temperature: IGN,
        max_temperature: 3.0,
        burning_rate: 0.0,
        ..StepConstants::new(K, H, 0.125)
    };
    let sources = Sources::new(&density_src, &heat)
        .with_fuel(&no_fuel)
        .with_solids(Solids {
            mask: &mask_f,
            velocity: &velocity,
        })
        .with_surface(&load_f);
    substep(
        &gpu,
        &mut cache,
        &mut pool,
        &mut state,
        sources,
        &c,
        PressureSolve::GaussSeidel(40),
    )
    .unwrap();
    let lost: f64 = state
        .surface
        .as_ref()
        .unwrap()
        .read_back(&gpu)
        .unwrap()
        .iter()
        .map(|&b| f64::from(b))
        .sum();
    let gained: f64 = state
        .fire
        .as_ref()
        .unwrap()
        .fuel
        .read_back(&gpu)
        .unwrap()
        .iter()
        .map(|&f| f64::from(f))
        .sum();
    assert!(lost > 0.0, "the hot cell must ignite the plank");
    assert!(
        (lost - gained).abs() <= 1e-3 * lost,
        "lost {lost}, gained {gained}"
    );
}

/// A closed cube mesh, [0.5, 1.5]³ in a 2 m domain, outward winding.
fn cube_mesh_collider(extra: &str) -> String {
    format!(
        r#"{{ "version": 3, "dims": [8, 8, 8], "fps": 24.0, "domain_size": 2.0,
      "nodes": [ {{ "id": 0, "kind": "ember.mesh_collider", "params": {{
        "mesh": {{ "positions": [[-0.5,-0.5,-0.5],[0.5,-0.5,-0.5],[0.5,0.5,-0.5],[-0.5,0.5,-0.5],
                               [-0.5,-0.5,0.5],[0.5,-0.5,0.5],[0.5,0.5,0.5],[-0.5,0.5,0.5]],
                  "indices": [0,2,1, 0,3,2, 4,5,6, 4,6,7, 0,1,5, 0,5,4,
                              1,2,6, 1,6,5, 2,3,7, 2,7,6, 3,0,4, 3,4,7] }},
        "transform": {{ "keys": [{{ "frame": 0, "translate": [1.0, 1.0, 1.0] }}{extra}] }},
        "surface_fuel": {{ "load": 2.5 }} }} }},
        {{ "id": 1, "kind": "core.output", "params": {{}} }} ],
      "edges": [ {{ "from_node": 0, "from_index": 2, "to_node": 1, "to_index": 0 }} ],
      "output": 1 }}"#
    )
}

#[test]
fn a_mesh_collider_outputs_its_load_and_rejects_a_moving_or_negative_one() {
    let loads = char_at_frame(&cube_mesh_collider(""), 1);
    let inside = loads.iter().filter(|&&l| l == 2.5).count();
    assert!(inside > 0, "the cube's cells carry the load");
    assert!(
        loads.iter().all(|&l| l == 0.0 || l == 2.5),
        "load or nothing"
    );

    let build = |doc: &str| {
        elements_core::graph::Document::from_json(doc)
            .unwrap()
            .into_graph(&elements_ember::registry())
            .map(|_| ())
    };
    let moving = cube_mesh_collider(r#",{ "frame": 10, "translate": [1.2, 1.0, 1.0] }"#);
    assert!(build(&moving).is_err(), "a keyframed mesh cannot burn");
    let negative = cube_mesh_collider("").replace("\"load\": 2.5", "\"load\": -1.0");
    assert!(build(&negative).is_err(), "a negative load is rejected");
}

// ---- The front spreads (FT4 spec §4 test 5) ----

/// A 16×16×32 domain (dx = 0.125 m, 2 × 2 × 4 m) with a vertical wooden wall
/// one cell thick (i = 8, j 5..=10, k 2..=13: `half_extents [0.0625, 0.4,
/// 0.8]` at [1.0625, 1.0, 1.0], load 4.0) and a temperature emitter at its
/// foot on the −x side ([0.9, 1.0, 0.3], r 0.15, rate `heat`). The solver
/// runs the preview preset with `temperature_dissipation` 8 and
/// `surface_burn_rate` 8 (attempt 2 in `docs/bench/surface-ignition.md`).
/// Fuel input 6 is connected at rate 0; `char` (solver output 4) is the output.
fn wall_doc(heat: f32) -> String {
    format!(
        r#"{{ "version": 3, "dims": [16, 16, 32], "fps": 24.0, "domain_size": 4.0,
      "nodes": [
        {{ "id": 0, "kind": "ember.sphere_emitter", "params": {{ "center": [0.9, 1.0, 0.3],
           "radius": 0.15, "density_rate": 0.0, "temperature_rate": {heat:?} }} }},
        {{ "id": 1, "kind": "ember.smoke_solver", "params": {{ "buoyancy_temperature": 1.0,
           "temperature_dissipation": 8.0, "surface_burn_rate": 8.0 }} }},
        {{ "id": 2, "kind": "core.output", "params": {{}} }},
        {{ "id": 3, "kind": "ember.sphere_emitter", "params": {{ "center": [0.5, 0.5, 3.5],
           "radius": 0.1, "density_rate": 0.0, "temperature_rate": 0.0 }} }},
        {{ "id": 4, "kind": "ember.collider", "params": {{
           "shape": {{ "box": {{ "half_extents": [0.0625, 0.4, 0.8] }} }},
           "transform": {{ "keys": [{{ "frame": 0, "translate": [1.0625, 1.0, 1.0] }}] }},
           "surface_fuel": {{ "load": 4.0 }} }} }} ],
      "edges": [
        {{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }},
        {{ "from_node": 0, "from_index": 1, "to_node": 1, "to_index": 1 }},
        {{ "from_node": 4, "from_index": 0, "to_node": 1, "to_index": 4 }},
        {{ "from_node": 4, "from_index": 1, "to_node": 1, "to_index": 5 }},
        {{ "from_node": 1, "from_index": 4, "to_node": 2, "to_index": 0 }},
        {{ "from_node": 3, "from_index": 0, "to_node": 1, "to_index": 6 }},
        {{ "from_node": 4, "from_index": 2, "to_node": 1, "to_index": 7 }} ],
      "output": 2 }}"#
    )
}

/// FT4 spec §4 test 5, emergent: heat at the foot of a vertical wall lights
/// the bottom rows, their fuel burns in the gas, and the flame rising along
/// the wall lights the rows above. Scene: `wall_doc(30.0)` (constants in its
/// doc comment): preview preset (one substep, MGPCG ×4), buoyancy 1,
/// temperature dissipation 8/s, surface burn rate 8/s, gas burning rate
/// 1.875, ignition 1.5, max temperature 3, frames 1..=60, 16×16×32, no seed
/// (nothing stochastic), recorded on Apple M1 Max. The zeroed-gather
/// mutant fails this test with `temperature_dissipation` 8 and
/// `surface_burn_rate` 8 changed together (not isolated from each other): the
/// heat source's own plume cools below ignition within two rows (the mutant
/// lit only rows 2 and 3), while flame is reset above it wherever fuel burns.
/// Every scene attempt is in
/// `docs/bench/surface-ignition.md`.
#[test]
fn the_front_spreads_up_a_wall() {
    const FRAMES: u32 = 60;
    const NZ: usize = 32;
    let mut s = Session::new(&wall_doc(30.0));
    let mut t = timeline(0);
    let mut first: [Option<u32>; NZ] = [None; NZ];
    for frame in 1..=FRAMES {
        let char: Vec<f32> = s
            .density_bits(&mut t, frame)
            .iter()
            .map(|b| f32::from_bits(*b))
            .collect();
        for (k, row) in char.chunks(16 * 16).enumerate() {
            if first[k].is_none() && row.iter().any(|&c| c > 0.0) {
                first[k] = Some(frame);
            }
        }
    }
    let rows: Vec<(usize, u32)> = first
        .iter()
        .enumerate()
        .filter_map(|(k, f)| f.map(|f| (k, f)))
        .collect();
    eprintln!("ignition frames by row: {rows:?}");
    assert!(
        rows.len() >= 3,
        "the front must reach at least 3 wall rows, got {rows:?}"
    );
    assert!(
        rows.windows(2).all(|w| w[1].1 >= w[0].1),
        "ignition frames must not decrease with height: {rows:?}"
    );
    assert!(
        rows[0].0 < rows.last().unwrap().0,
        "the front must have moved"
    );
}
