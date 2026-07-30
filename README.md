# Rainbow FMM

A million-body rainbow gravity field, planned in Rust, exported as a small
WebAssembly module, and executed by WebGPU compute shaders.

The default preset is `1,048,576` particles and is sized for an overclocked
RX 6800. Smaller presets are useful on laptop and software adapters.

## Why the BLDTT paper

This project takes its GPU structure from Wilson, Vaughn, and Krasny's
[GPU-accelerated FMM based on barycentric Lagrange interpolation and dual tree
traversal](https://doi.org/10.1016/j.cpc.2021.108017), rather than the newer
[jaxFMM](https://arxiv.org/abs/2511.15269).

The choice is practical:

- BLDTT demonstrates O(N) scaling from 100 thousand to 100 million particles.
- Its target-outer/source-inner kernels and upward/downward passes are natural
  WebGPU workloads.
- Its dual-tree idea supports non-uniform refinement without making particles
  perform independent O(log N) tree walks.
- jaxFMM's median-split hierarchy costs O(N log N) to build, and its current
  interaction-list construction is dynamic and not JIT compiled. Its
  rotation-based spherical-harmonic translations are also a poor match for a
  compact first WGSL implementation.

This is not a line-by-line port of BLDTT. Barycentric proxy grids would be too
expensive for an interactive million-body force update in a browser. The demo
uses a low-order (`p = 1`) Cartesian expansion:

1. P2M deposits fixed-point mass and first moments into the finest grid.
2. M2M builds a complete linear octree with eight-child reductions.
3. Cells with at most 16 particles terminate locally as adaptive leaves.
4. M2L evaluates the bounded 189-cell interaction stencil only along active
   density paths, accumulating acceleration and a symmetric tidal tensor.
5. L2L translates those local fields down the active hierarchy.
6. Adaptive leaves evaluate their 27-cell near neighborhood with softened
   monopoles; each particle samples its leaf field.

The finest allocated grid is bounded at depth six. Empty cells exit after a
single count check, and the depth is a constant for the supported range.
Allocated cells are proportional to the selected particle budget (about four
particles per finest cell), so the complete step is

```text
P2M O(N) + M2M O(C) + M2L O(189 C) + L2L O(C) + sampling O(N)
```

with `C = O(N)`, hence O(N). There is no all-pairs fallback.

## Build and run

Install a current Rust toolchain with the `wasm32-unknown-unknown` target, then:

```bash
./build-web.sh
./serve.sh 8080
```

Open <http://127.0.0.1:8080>. A local HTTP origin is required for WebGPU and
Wasm streaming.

The Wasm module is dependency-free. Rust owns octree sizing, the bounded work
model, public ABI, and the embedded WGSL payload. The browser module is a thin
WebGPU command scheduler and UI layer.

### Interactive shepherd

Select a tool in the simulation deck, or press its number key:

1. **Orbit** — drag the camera; the wheel zooms.
2. **Pull** — hold and drag a localized attractor.
3. **Push** — hold and drag a localized repulsor.
4. **Vortex** — hold and drag to add angular momentum around the camera-facing
   axis.

Field force and radius are adjustable. Each active tool evaluates one smooth,
bounded field per particle, so interaction adds O(N) work and does not alter
the solver's asymptotic complexity.

## Memory at one million particles

| Buffer | Approximate size |
|---|---:|
| Positions, dynamical hue, velocities, excitation | 32 MiB |
| 299,593 fixed-point cell moments | 4.6 MiB |
| Acceleration + tidal local fields | 13.7 MiB |
| Half-scale HDR field + blur target at 1080p | about 8 MiB |
| Uniform plans and overhead | < 1 MiB |
| **Solver + visual field total** | **about 59 MiB** |

This leaves substantial headroom in a 16 GiB RX 6800 for the browser,
swapchain, and driver allocations. Actual frame rate depends on browser,
resolution, clock/power limits, and particle distribution; use the live frame
readout rather than treating a build-time estimate as a benchmark.

All 1,048,576 bodies participate in both the force solve and the visual pass.
Each body makes one additive deposit into a half-scale `rgba16float`
color-and-density field. A fixed horizontal/vertical Gaussian reconstruction
then expands that compact field into large, smooth gradients. The fullscreen
resolve divides accumulated color by accumulated density before tone mapping,
so overlap increases presence without bleaching every structure into fog. No
visual subsample is hidden; the footer reports the full visible population.
The **Gradient radius** control is expressed in final-canvas pixels and changes
only the reconstruction radius; it no longer changes exposure at the same
time.

The seed uses a faint live spherical component as a self-consistent orbital
backbone, plus a luminous four-arm disc and a counter-rotating polar annulus.
The FMM domain spans `[-1.60, +1.60]` on each axis—4.1 times the volume of the
original unit half-extent—while the central seed retains its original scale.
The halo bodies move and contribute to gravity; their low luminosity prevents
them from turning the view into fog. Luminous bodies receive a continuous hue
from their initial velocity direction, orbital inclination, radial motion,
excitation, and a small lineage phase. That hue persists as material state and
is slowly advected by shocks and interactive work, producing gradients that
flow around an orbit rather than discrete lane colors. Absolute position is
not a color input.

In local Playwright runs on 2026-07-30, Chromium exposed an AMD `rdna-2`
adapter (but no model string). With one clean browser process, the current
million-deposit HDR build held 60 FPS at both 12 and 30 seconds in a 1920×1080
viewport with a 1885×1060 internal canvas. Earlier 50–53 FPS readings were
contaminated by Chrome processes left alive between Playwright runs. The live
frame loop is deliberately not queue-drained: a per-frame
`onSubmittedWorkDone()` fence starved Firefox's submit/present pipeline. A
8,192-particle readback after exercising Pull, Push, and Vortex contained no
non-finite values. These are measurements from one machine, not a portable RX
6800 guarantee.

## Physics and energy

The force step is softened and approximate. Symplectic Euler keeps energy
error bounded for a smooth Hamiltonian, but the p=1 cell approximation,
density changes, and emergency escape boundary mean this demo does **not**
claim exact energy conservation.

Pull, Push, and Vortex are explicitly external fields. While held, the status
line says `EXTERNAL WORK`: they deliberately exchange energy or angular
momentum with the simulation. Releasing the pointer removes the field and
returns to the unforced symplectic evolution.

### Optional a-life thought experiment

A conservative artificial-life version can be interesting if "organisms" are
not implemented as free energy appearing from rules:

- Give every particle an internal oscillator coordinate and conjugate momentum.
  Hue visualizes oscillator phase rather than species ID.
- Make bonds continuous double-well potentials. Forming a bond releases
  potential energy into translation, rotation, or internal oscillation;
  breaking it must absorb the same amount.
- Use FMM for the long-range field and the adaptive leaf lists for short-range
  conservative chemistry.
- Express metabolism as energy transfer from explicit fuel particles or a
  finite field reservoir. Reproduction redistributes mass and energy instead
  of creating either.
- Integrate positions, internal coordinates, and bonds with a reversible
  splitting method, then track total Hamiltonian, linear momentum, and angular
  momentum with compensated GPU reductions.

That could produce selection and self-maintaining structures while preserving
energy. It is deliberately not mixed into this visual gravity demo: reaction
topology, conservative bond switching, and energy accounting deserve their own
verification rather than a decorative toggle.

## Validation

```bash
cargo test
```

Browser validation uses the checked-in Playwright CLI configuration:

```bash
playwright-cli -s=rainbow open 'http://127.0.0.1:8080/?particles=131072'
playwright-cli -s=rainbow console
```

For an intentionally small diagnostic plan, pass `?particles=16384`. The
public control starts at 131,072 and tops out at 1,048,576.
