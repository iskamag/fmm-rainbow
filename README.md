# Rainbow FMM

A million-body rainbow gravity field, planned in Rust, exported as a small
WebAssembly module, and executed by WebGPU compute shaders.

The default preset is `1,048,576` particles and is sized for an overclocked
RX 6800. The budget reaches up to `5,242,880` (5M); smaller presets are useful
on laptop and software adapters.

## Why the BLDTT paper

This project takes its GPU structure from Wilson, Vaughn, and Krasny's
[GPU-accelerated FMM based on barycentric Lagrange interpolation and dual tree
traversal](https://doi.org/10.1016/j.cpc.2021.108017), rather than the newer
[jaxFMM](https://arxiv.org/abs/2511.15269).

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

### Native GPU harness

`native/` is a standalone wgpu harness that runs the same WGSL payload and
planner headlessly on the GPU, with no browser or display. It reports the
adapter, a content digest of the field and moment buffers (for cross-checking
against the browser), per-phase GPU time from timestamp queries, the amortized
wall-clock cost of a solve, and a finiteness check:

```bash
cd native
cargo run --release -- 1048576 60
```

