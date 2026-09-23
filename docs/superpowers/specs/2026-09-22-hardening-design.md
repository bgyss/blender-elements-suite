# Core Hardening Before Piece 2b — Design

**Date:** 2026-09-22
**Status:** Approved for planning
**Addresses:** risks (a), (g) and (i) in §6 of `2026-09-21-ember-solver-design.md`

Piece 2b will add bigger bakes and a benchmark that compares totals. Three things
found in 2a's final review would undermine both. This slice fixes them before 2b
is designed. It is independent of 2b's solver work.

## 1. Request the adapter's buffer limit (risk a)

`GpuContext` asks for `Limits::downlevel_defaults()` with only the resolution
limits raised to the adapter's. So `max_buffer_size` stays at 256 MiB, and
`Document::validate_for` rejects any domain above about 406³, although the
umbrella promises 512³ offline bakes. The M1 Max reports 4095 MiB.

- A pure function, `gpu::required_limits(adapter: &wgpu::Limits) -> wgpu::Limits`,
  returns `downlevel_defaults()` with the resolution limits **and**
  `max_buffer_size` taken from the adapter. Nothing else is raised. In
  particular, the 4-storage-textures-per-stage limit stays, so kernels remain
  portable.
- This is a limit, not a feature: `required_features` stays empty.
- Read-back in slabs, for devices with small limits, stays with piece 3's export work.
- The daemon test that asserts a 512³ document is rejected at load now uses
  2047³ (about 34 GB). That exceeds any adapter seen so far and stays under
  the 2048 texture limit, and the test still checks that the error names the size.

## 2. Out-of-memory errors don't crash the engine (risk g)

wgpu sends an error that no matching error scope catches to the device's
uncaptured-error handler, and with none installed, to `default_error_handler`,
which **panics** (`wgpu-30.0.1/src/backend/wgpu_core.rs`). `GpuContext::scoped`
pushes only a Validation scope, so an out-of-memory or internal error panics the
daemon, the failure the out-of-process design exists to prevent.

**Core**
- `GpuError` gains `OutOfMemory(String)` and `Internal(String)`.
- `scoped` pushes OutOfMemory, Internal and Validation scopes, and pops them in
  reverse order.
- A pure function decides what `scoped` reports when more than one thing went
  wrong, in this order: device lost (keeping the detail of anything else
  captured), then out of memory, then internal, then validation, then an error
  raised earlier outside any scope.
- `GpuContext` installs an uncaptured-error handler when the device is created.
  It records the first error raised outside any scope, where today that error
  would panic, and the next `scoped` call reports it. The docs say plainly that
  such an error may be attributed to that later, unrelated call.
- `FieldPool::clear()` drops every pooled texture, so memory can actually be
  given back.

**Daemon and protocol**
- A new `ErrorKind::OutOfMemory`, which serializes as `out_of_memory`. Mapping
  it to the existing `gpu` kind would leave the add-on repeating a render that
  cannot succeed.
- On `out_of_memory` the daemon resets the timeline (dropping cached snapshots
  and state) and clears the pool.
- `ELEMENTS_PROTOCOL_VERSION` and the add-on's `PROTOCOL_VERSION` go from 1 to
  2, because the protocol's rule is "bumped whenever a message changes shape".

**Add-on**
- A bpy-free `errors.describe(kind, message)` returns the status line and
  whether Live mode must stop. On `out_of_memory` it says to lower the
  resolution and stops Live mode, so the viewport stops re-rendering into the
  same failure.

**Limit, stated in the docs:** this covers what wgpu reports. On Apple
Silicon's unified memory the OS may swap, or end the process, before wgpu
reports anything, and no error scope can catch that.

## 3. A mass test that notices swapped wiring (risk i)

No test fails if the solver's density and temperature are swapped, at its
inputs or its outputs. The Mantaflow benchmark will compare totals, so this must
be pinned first.

- One frame of the solver through real graphs, with buoyancy off, so nothing
  moves and advection returns its input exactly. The emitter's rates differ
  (2 for density, 5 for temperature).
- Solver output 0's total must equal the emitter's density-source total × dt, and
  output 1's the temperature-source total × dt, each within 1e-5 relative.

## Testing

Every new test is proven to fail under one single-change mutation. A real
out-of-memory error cannot be forced reliably, so:
- the error sorting and precedence are unit-tested with constructed `wgpu::Error`s;
- the uncaptured-error handler is tested with a real validation error raised
  outside any scope, which panics today;
- the daemon's reset-and-clear on `out_of_memory` is not exercised by a test.
  That is recorded as a known gap.
