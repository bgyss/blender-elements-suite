//! Shapes and keyframed transforms, shared by emitters and colliders
//! (2b-2 spec §2.1). The CPU evaluates the keys into a `Pose`; the GPU only
//! reads the resulting matrix and velocities.

use elements_core::graph::DocError;
use serde::{Deserialize, Serialize};

use crate::params;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", deny_unknown_fields)]
pub enum Shape {
    Sphere { radius: f32 },
    Box { half_extents: [f32; 3] },
}

impl Shape {
    pub fn validate(&self, kind: &str) -> Result<(), DocError> {
        match self {
            Self::Sphere { radius } => {
                params::finite(kind, "radius", &[*radius])?;
                if *radius <= 0.0 {
                    return Err(params::bad(
                        kind,
                        format!("radius must be positive, got {radius}"),
                    ));
                }
            }
            Self::Box { half_extents } => {
                params::finite(kind, "half_extents", half_extents)?;
                if half_extents.iter().any(|&e| e <= 0.0) {
                    return Err(params::bad(
                        kind,
                        format!("half_extents must be positive, got {half_extents:?}"),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rotate {
    pub axis: [f32; 3],
    pub degrees: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Key {
    pub frame: f32,
    #[serde(default)]
    pub translate: [f32; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotate: Option<Rotate>,
}

/// A keyframed rigid transform (spec §2.1): translation interpolates
/// linearly between keys, and rotation by slerp.
///
/// Rotation between consecutive keys takes the shorter arc, so each key's
/// orientation must be less than 180° from the previous one's. A full spin
/// therefore needs at least three keys; two keys 360° apart are the same
/// orientation and do not rotate at all.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Transform {
    pub keys: Vec<Key>,
}

/// Where an object is at one moment and how it moves, in world space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    /// Local-to-world rotation as a unit quaternion, [w, x, y, z].
    pub rotation: [f64; 4],
    pub translate: [f64; 3],
    /// dT/dt, m/s.
    pub linear: [f64; 3],
    /// ω, rad/s.
    pub angular: [f64; 3],
}

type Quat = [f64; 4];

fn q_mul(a: Quat, b: Quat) -> Quat {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn q_conj(q: Quat) -> Quat {
    [q[0], -q[1], -q[2], -q[3]]
}

fn q_dot(a: Quat, b: Quat) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

fn q_axis_angle(r: Option<Rotate>) -> Quat {
    let Some(r) = r else {
        return [1.0, 0.0, 0.0, 0.0];
    };
    let a = r.axis.map(f64::from);
    let len = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let half = f64::from(r.degrees).to_radians() * 0.5;
    let s = half.sin() / len;
    [half.cos(), a[0] * s, a[1] * s, a[2] * s]
}

/// Spherical interpolation along the shorter arc.
fn q_slerp(a: Quat, b: Quat, t: f64) -> Quat {
    let mut b = b;
    let mut d = q_dot(a, b);
    if d < 0.0 {
        b = b.map(|x| -x);
        d = -d;
    }
    if d > 1.0 - 1e-12 {
        return a;
    }
    let theta = d.clamp(-1.0, 1.0).acos();
    let (s0, s1) = (((1.0 - t) * theta).sin(), (t * theta).sin());
    let inv = 1.0 / theta.sin();
    std::array::from_fn(|i| (a[i] * s0 + b[i] * s1) * inv)
}

/// The rotation taking `a` to `b`, as an axis times an angle (world frame),
/// along the shorter arc.
fn q_rotation_vector(a: Quat, b: Quat) -> [f64; 3] {
    let mut r = q_mul(b, q_conj(a));
    if r[0] < 0.0 {
        r = r.map(|x| -x);
    }
    let s = (r[1] * r[1] + r[2] * r[2] + r[3] * r[3]).sqrt();
    if s < 1e-15 {
        return [0.0; 3];
    }
    let angle = 2.0 * s.atan2(r[0]);
    [r[1] / s * angle, r[2] / s * angle, r[3] / s * angle]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

impl Transform {
    /// One key at frame 0: a static object at `translate`.
    pub fn at(translate: [f32; 3]) -> Self {
        Self {
            keys: vec![Key {
                frame: 0.0,
                translate,
                rotate: None,
            }],
        }
    }

    pub fn validate(&self, kind: &str) -> Result<(), DocError> {
        if self.keys.is_empty() {
            return Err(params::bad(kind, "transform needs at least one key"));
        }
        for key in &self.keys {
            params::finite(kind, "key frame", &[key.frame])?;
            params::finite(kind, "translate", &key.translate)?;
            if let Some(r) = key.rotate {
                params::finite(
                    kind,
                    "rotate",
                    &[r.axis[0], r.axis[1], r.axis[2], r.degrees],
                )?;
                let len2 = r
                    .axis
                    .iter()
                    .map(|&a| f64::from(a) * f64::from(a))
                    .sum::<f64>();
                if len2 < 1e-24 {
                    return Err(params::bad(kind, "rotate axis must be nonzero"));
                }
            }
        }
        if self.keys.windows(2).any(|w| w[1].frame <= w[0].frame) {
            return Err(params::bad(kind, "key frames must strictly increase"));
        }
        Ok(())
    }

    /// The pose at `frame` (it may be fractional). Velocities are per second,
    /// with `seconds_per_frame` converting the key spacing. Before the first
    /// key and from the last key on, the transform holds with zero velocity.
    /// Between keys, rotation takes the shorter arc (see [`Transform`]).
    ///
    /// Requires a transform that passed [`Transform::validate`]: it indexes
    /// `keys[0]`, so it panics on an empty key list, and its `expect` on the
    /// bracketing key segment assumes frames in strictly increasing order.
    pub fn pose(&self, frame: f64, seconds_per_frame: f64) -> Pose {
        let hold = |k: &Key| Pose {
            rotation: q_axis_angle(k.rotate),
            translate: k.translate.map(f64::from),
            linear: [0.0; 3],
            angular: [0.0; 3],
        };
        let first = &self.keys[0];
        let last = &self.keys[self.keys.len() - 1];
        if frame < f64::from(first.frame) || self.keys.len() == 1 {
            return hold(first);
        }
        if frame >= f64::from(last.frame) {
            return hold(last);
        }
        let i = self
            .keys
            .windows(2)
            .position(|w| frame >= f64::from(w[0].frame) && frame < f64::from(w[1].frame))
            .expect("frame lies strictly inside the keyed range");
        let (k0, k1) = (&self.keys[i], &self.keys[i + 1]);
        let (f0, f1) = (f64::from(k0.frame), f64::from(k1.frame));
        let t = (frame - f0) / (f1 - f0);
        let seconds = (f1 - f0) * seconds_per_frame;
        let (t0, t1) = (k0.translate.map(f64::from), k1.translate.map(f64::from));
        let (q0, q1) = (q_axis_angle(k0.rotate), q_axis_angle(k1.rotate));
        Pose {
            rotation: q_slerp(q0, q1, t),
            translate: std::array::from_fn(|a| t0[a] + (t1[a] - t0[a]) * t),
            linear: std::array::from_fn(|a| (t1[a] - t0[a]) / seconds),
            angular: q_rotation_vector(q0, q1).map(|w| w / seconds),
        }
    }
}

impl Pose {
    /// World-to-local rotation, row-major: the transpose of local-to-world.
    pub fn world_to_local(&self) -> [[f64; 3]; 3] {
        let [w, x, y, z] = self.rotation;
        // Local-to-world matrix of a unit quaternion, then transposed.
        let m = [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ];
        std::array::from_fn(|r| std::array::from_fn(|c| m[c][r]))
    }

    /// Velocity of the object's material at world point `x`, m/s.
    pub fn velocity_at(&self, x: [f64; 3]) -> [f64; 3] {
        let r = std::array::from_fn(|a| x[a] - self.translate[a]);
        let spin = cross(self.angular, r);
        std::array::from_fn(|a| self.linear[a] + spin[a])
    }

    /// The rotation's angle about `axis` (unit length), for tests.
    pub fn angle_about(&self, axis: [f64; 3]) -> f64 {
        let [w, x, y, z] = self.rotation;
        let along = x * axis[0] + y * axis[1] + z * axis[2];
        2.0 * along.atan2(w)
    }
}

/// Matches `Shape` in `shape.wgsl`: a `mat3x3<f32>` (three 16-byte columns),
/// then four vec3 + scalar rows. 112 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ShapeGpu {
    world_to_local: [[f32; 4]; 3],
    origin: [f32; 3],
    kind: u32,
    extents: [f32; 3],
    _pad0: u32,
    linear: [f32; 3],
    _pad1: u32,
    angular: [f32; 3],
    _pad2: u32,
}

const _: () = assert!(std::mem::size_of::<ShapeGpu>() == 112);

impl ShapeGpu {
    pub(crate) fn new(shape: &Shape, pose: &Pose) -> Self {
        let m = pose.world_to_local();
        let (kind, extents) = match *shape {
            Shape::Sphere { radius } => (0, [radius, 0.0, 0.0]),
            Shape::Box { half_extents } => (1, half_extents),
        };
        Self {
            // WGSL matrices are column-major: column c holds row r's entry c.
            world_to_local: std::array::from_fn(|c| {
                [m[0][c] as f32, m[1][c] as f32, m[2][c] as f32, 0.0]
            }),
            origin: pose.translate.map(|v| v as f32),
            kind,
            extents,
            _pad0: 0,
            linear: pose.linear.map(|v| v as f32),
            _pad1: 0,
            angular: pose.angular.map(|v| v as f32),
            _pad2: 0,
        }
    }
}
