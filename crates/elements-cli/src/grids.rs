//! What `bake` writes per frame when a document names its outputs.

use anyhow::{Context, bail, ensure};
use elements_core::gpu::{Axis, FieldDims, StaggeredField};
use elements_core::graph::{DocOutput, Graph, NodeId, SocketType};

/// Average each cell's two faces along each axis (spec §3.3).
pub fn cell_centred_velocity(cells: [u32; 3], faces: [&[f32]; 3]) -> anyhow::Result<[Vec<f32>; 3]> {
    let [nx, ny, nz] = cells.map(|c| c as usize);
    let cell_dims = FieldDims::new(cells[0], cells[1], cells[2]);
    let mut out = [Vec::new(), Vec::new(), Vec::new()];
    for (a, axis) in Axis::ALL.into_iter().enumerate() {
        let fd = StaggeredField::face_dims(cell_dims, axis);
        let (fx, fy) = (fd.x as usize, fd.y as usize);
        ensure!(
            faces[a].len() == fx * fy * fd.z as usize,
            "{axis:?} face has {} values, expected {}",
            faces[a].len(),
            fx * fy * fd.z as usize
        );
        let at = |x: usize, y: usize, z: usize| faces[a][(z * fy + y) * fx + x];
        let mut v = Vec::with_capacity(nx * ny * nz);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let (lo, hi) = match axis {
                        Axis::X => (at(x, y, z), at(x + 1, y, z)),
                        Axis::Y => (at(x, y, z), at(x, y + 1, z)),
                        Axis::Z => (at(x, y, z), at(x, y, z + 1)),
                    };
                    v.push(0.5 * (lo + hi));
                }
            }
        }
        out[a] = v;
    }
    Ok(out)
}

/// The final grid names, in file order; a vector entry `n` becomes `n_x`, `n_y`, `n_z`.
pub fn expanded_names(graph: &Graph, outputs: &[DocOutput]) -> anyhow::Result<Vec<String>> {
    let mut names: Vec<String> = Vec::new();
    for entry in outputs {
        let spec = graph.node(NodeId(entry.node))?.sockets();
        let ty = *spec
            .outputs
            .get(entry.socket as usize)
            .with_context(|| format!("output {:?} has no socket {}", entry.name, entry.socket))?;
        let new: Vec<String> = match ty {
            SocketType::Field => vec![entry.name.clone()],
            SocketType::VectorField => ["x", "y", "z"]
                .iter()
                .map(|s| format!("{}_{s}", entry.name))
                .collect(),
            other => bail!("output {:?} is a {other:?}, not a field", entry.name),
        };
        for name in new {
            ensure!(
                !names.contains(&name),
                "two outputs would both write a grid named {name:?}"
            );
            names.push(name);
        }
    }
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::{cell_centred_velocity, expanded_names};
    use elements_core::graph::Document;

    #[test]
    fn velocity_is_the_mean_of_each_cells_two_faces() {
        // 2x1x1 cells: x faces are 3x1x1, y faces 2x2x1, z faces 2x1x2. Face index is
        // (z * face_ny + y) * face_nx + x, so the y faces are [y=0: 10, 20 | y=1: 30, 50].
        let fx = [1.0, 3.0, 7.0];
        let fy = [10.0, 20.0, 30.0, 50.0];
        let fz = [100.0, 200.0, 300.0, 500.0];
        let [vx, vy, vz] = cell_centred_velocity([2, 1, 1], [&fx, &fy, &fz]).unwrap();
        assert_eq!(vx, [2.0, 5.0]);
        assert_eq!(vy, [20.0, 35.0]);
        assert_eq!(vz, [200.0, 350.0]);
    }

    #[test]
    fn a_face_array_of_the_wrong_length_is_an_error() {
        let fx = [0.0; 3];
        let short = [0.0; 3];
        assert!(cell_centred_velocity([2, 1, 1], [&fx, &short, &short]).is_err());
    }

    fn solver_doc(outputs: &str) -> String {
        format!(
            r#"{{
              "version": 4, "dims": [8, 8, 8],
              "nodes": [
                {{ "id": 0, "kind": "ember.emitter", "params": {{
                  "shape": {{ "sphere": {{ "radius": 0.1 }} }},
                  "transform": {{ "keys": [{{ "frame": 0 }}] }} }} }},
                {{ "id": 1, "kind": "ember.smoke_solver", "params": {{}} }},
                {{ "id": 2, "kind": "core.output", "params": {{}} }}
              ],
              "edges": [
                {{ "from_node": 0, "from_index": 0, "to_node": 1, "to_index": 0 }},
                {{ "from_node": 1, "from_index": 0, "to_node": 2, "to_index": 0 }}
              ],
              "output": 2,
              "outputs": {outputs}
            }}"#
        )
    }

    fn names_for(outputs: &str) -> anyhow::Result<Vec<String>> {
        let doc = Document::from_json(&solver_doc(outputs)).unwrap();
        let outs = doc.outputs.clone();
        let (graph, _) = doc.into_graph(&elements_ember::registry()).unwrap();
        expanded_names(&graph, &outs)
    }

    #[test]
    fn a_vector_output_expands_to_three_grids() {
        let names = names_for(
            r#"[{ "node": 1, "socket": 0, "name": "density" },
                { "node": 1, "socket": 2, "name": "velocity" }]"#,
        )
        .unwrap();
        assert_eq!(names, ["density", "velocity_x", "velocity_y", "velocity_z"]);
    }

    #[test]
    fn a_scalar_named_like_an_expanded_component_is_an_error() {
        let err = names_for(
            r#"[{ "node": 1, "socket": 2, "name": "velocity" },
                { "node": 1, "socket": 0, "name": "velocity_x" }]"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("velocity_x"), "{err}");
    }
}
