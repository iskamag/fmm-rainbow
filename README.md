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

## Memory at one million particles

| Buffer | Approximate size |
|---|---:|
| Positions, masses, velocities, hue | 32 MiB |
| 299,593 fixed-point cell moments | 4.6 MiB |
| Acceleration + tidal local fields | 13.7 MiB |
| Uniform plans and overhead | < 1 MiB |
| **Core solver total** | **about 51 MiB** |

This leaves substantial headroom in a 16 GiB RX 6800 for the browser,
swapchain, and driver allocations. Actual frame rate depends on browser,
resolution, clock/power limits, and particle distribution; use the live frame
readout rather than treating a build-time estimate as a benchmark.

All 1,048,576 bodies participate in the force solve. The visual pass draws a
deterministically distributed 262,144-body sample and caps its internal canvas
near two million pixels. This avoids spending most of the frame on six million
blended vertices at 4K/Retina resolution; the footer reports the visible
sample independently from the simulated population.

In local Playwright runs on 2026-07-30, Chromium exposed an AMD `rdna-2`
adapter (but no model string). Queue-drained diagnostics measured 7.3 ms at a
1920×1080 viewport and 8.6 ms at a 3840×2160 viewport. The live frame loop is
deliberately not queue-drained: a per-frame `onSubmittedWorkDone()` fence
starved Firefox's submit/present pipeline. With that fence removed, the full
million-particle loop held the test display's 60 Hz ceiling at 1280×720. A
4,096-particle readback contained no non-finite values. These are measurements
from one machine, not a portable RX 6800 guarantee.

## Physics and energy

The force step is softened and approximate. Symplectic Euler keeps energy
error bounded for a smooth Hamiltonian, but the p=1 cell approximation,
density changes, and emergency escape boundary mean this demo does **not**
claim exact energy conservation.

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
