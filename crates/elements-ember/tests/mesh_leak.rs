mod common;

use common::*;
use elements_core::gpu::{FieldDims, FieldFormat, FieldPool, PipelineCache};
use elements_ember::collider::ColliderFields;
use elements_ember::emitter::{Sphere, fill_sphere};
use elements_ember::kernels::{Solids, StepConstants};
use elements_ember::mesh::Mesh;
use elements_ember::mesh_collider::{MeshColliderParams, fill_mesh_collider};
use elements_ember::solver::{PressureSolve, SolverState, Sources, substep};
use elements_ember::transform::Transform;

const DX: f32 = 2.0 / 32.0;
const FRAMES: usize = 80;

/// Mean density above a horizontal 0.03 m wall (cells with z > 1.2) after 80
/// frames of a plume rising into it from z = 0.3, at 32³ in a 2 m domain.
/// (40 frames is too few: the plume front has not reached z = 1.2.)
/// The wall is thinner than a voxel, so `offset` decides whether it seals.
fn above_wall(offset: f32, with_wall: bool) -> f64 {
    let gpu = gpu();
    let mut pool = FieldPool::new();
    let mut cache = PipelineCache::new();
    let cells = FieldDims::new(32, 32, 32);
    let ds = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    let ts = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
    let sphere = Sphere {
        center: [1.0, 1.0, 0.3],
        radius: 0.2,
        density_rate: 1.0,
        temperature_rate: 2.0,
    };
    fill_sphere(&gpu, &mut cache, &ds, &ts, &sphere, DX).unwrap();

    let wall = with_wall.then(|| {
        let params = MeshColliderParams {
            mesh: Mesh::box_mesh([-1.0, -1.0, -0.015], [1.0, 1.0, 0.015]),
            transform: Transform::at([1.0, 1.0, 1.0]),
            offset,
            surface_fuel: None,
        };
        let sdf = pool.acquire(&gpu, cells, FieldFormat::R32Float).unwrap();
        // Static wall: the face velocity stays zero.
        let velocity = pool.acquire_staggered_uninit(&gpu, cells).unwrap();
        let pose = params.transform.pose(0.0, 1.0 / 24.0);
        fill_mesh_collider(
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
        let mask = solid_mask(&gpu, &mut cache, &mut pool, cells, DX, &sdf);
        (sdf, mask, velocity)
    });
    let mut sources = Sources::new(&ds, &ts);
    if let Some((_, mask, velocity)) = &wall {
        sources = sources.with_solids(Solids { mask, velocity });
    }
    let constants = StepConstants {
        beta: 1.0,
        has_solids: with_wall,
        ..StepConstants::new(cells, 1.0 / 24.0, DX)
    };
    let mut state = SolverState::zeroed(&gpu, &mut cache, &mut pool, cells).unwrap();
    for _ in 0..FRAMES {
        substep(
            &gpu,
            &mut cache,
            &mut pool,
            &mut state,
            sources,
            &constants,
            PressureSolve::GaussSeidel(160),
        )
        .unwrap();
    }
    let density = state.density.read_back(&gpu).unwrap();
    if let Some((sdf, _, _)) = &wall {
        let sdf = sdf.read_back(&gpu).unwrap();
        let (i, j) = (16, 16);
        // Pin the hand-derived geometry: both layers of cell centres (k = 15
        // and 16) are 0.01625 outside the wall's faces, less `offset`.
        for k in [15, 16] {
            let got = sdf[index(cells, i, j, k)];
            let want = 0.01625 - offset;
            assert!(
                (got - want).abs() < 1e-4,
                "sdf at k={k}: {got}, want {want}"
            );
        }
    }
    let (mut sum, mut n) = (0.0f64, 0.0f64);
    for k in 0..cells.z {
        for j in 0..cells.y {
            for i in 0..cells.x {
                if (f64::from(k) + 0.5) * f64::from(DX) > 1.2 {
                    sum += f64::from(density[index(cells, i, j, k)]);
                    n += 1.0;
                }
            }
        }
    }
    sum / n
}

/// Review focus 3: a plank thinner than one voxel blocks flow only when
/// `offset` inflates it. Both z layers of cell centres sit 0.03125 from the
/// wall's mid-plane, 0.01625 outside its faces, so at offset 0 the SDF is
/// positive there and the solver sees no solid; offset = dx/2 makes it
/// negative at both layers.
#[test]
fn a_plank_thinner_than_a_voxel_blocks_the_plume_only_with_an_offset() {
    let open = above_wall(0.0, false);
    let leaky = above_wall(0.0, true);
    let sealed = above_wall(0.5 * DX, true);
    eprintln!("open {open}, leaky {leaky}, sealed {sealed}");
    assert!(
        open > 1e-3,
        "control: the open plume reaches z>1.2 ({open})"
    );
    assert!(
        leaky > 0.2 * open,
        "without an offset the thin plank leaks ({leaky} vs {open})"
    );
    assert!(
        sealed < 0.01 * open,
        "with offset = half a voxel it seals ({sealed} vs {open})"
    );
}
