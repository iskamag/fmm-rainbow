const $ = (selector) => document.querySelector(selector);

const ui = {
  canvas: $("#universe"),
  count: $("#particle-count"),
  gravity: $("#gravity"),
  gravityValue: $("#gravity-value"),
  softening: $("#softening"),
  softeningValue: $("#softening-value"),
  speed: $("#speed"),
  speedValue: $("#speed-value"),
  glow: $("#glow"),
  glowValue: $("#glow-value"),
  toggle: $("#toggle-run"),
  reseed: $("#reseed"),
  collapse: $("#collapse-controls"),
  deck: $(".control-deck"),
  fullscreen: $("#fullscreen"),
  particleStat: $("#particle-stat"),
  fpsStat: $("#fps-stat"),
  frameStat: $("#frame-stat"),
  runState: $("#run-state"),
  gpuName: $("#gpu-name"),
  treeStat: $("#tree-stat"),
  errorPanel: $("#error-panel"),
  errorMessage: $("#error-message"),
};

const state = {
  wasm: null,
  shader: "",
  algorithm: "",
  adapter: null,
  device: null,
  context: null,
  format: null,
  pipelines: {},
  buffers: null,
  computeBindGroup: null,
  renderBindGroup: null,
  computeParams: null,
  renderParams: null,
  count: 1 << 20,
  depth: 6,
  cells: 299_593,
  renderCount: 262_144,
  renderStride: 4,
  resolutionScale: 1,
  running: true,
  ready: false,
  rebuilding: false,
  yaw: -0.18,
  pitch: -0.14,
  zoom: 2.32,
  dragging: false,
  pointerX: 0,
  pointerY: 0,
  elapsed: 0,
  frameSamples: [],
  lastFrameAt: performance.now(),
  submittedFrames: 0,
};

const PARAM_BYTES = 80;
const PARAM_STRIDE = 256;
const PARAM_SLOTS = 16;
const PARTICLE_BYTES = 32;
const MOMENT_BYTES = 16;
const FIELD_BYTES = 48;

async function instantiateWasm(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Rust/Wasm payload returned HTTP ${response.status}`);
  }
  try {
    return (await WebAssembly.instantiateStreaming(response, {})).instance;
  } catch {
    const bytes = await (await fetch(url)).arrayBuffer();
    return (await WebAssembly.instantiate(bytes, {})).instance;
  }
}

function wasmString(ptr, length) {
  return new TextDecoder().decode(
    new Uint8Array(state.wasm.exports.memory.buffer, ptr, length),
  );
}

function levelOffset(level) {
  return state.wasm.exports.rainbow_level_offset(level);
}

function levelCells(level) {
  return state.wasm.exports.rainbow_level_cells(level);
}

function makeParams(level, upward = false, render = false) {
  const data = new ArrayBuffer(PARAM_BYTES);
  const view = new DataView(data);
  const levelDim = 1 << level;
  const parentLevel = upward ? level + 1 : Math.max(0, level - 1);
  const parentDim = 1 << parentLevel;
  const uints = [
    state.count,
    render ? state.renderStride : state.depth,
    render ? state.renderCount : level,
    levelOffset(level),
    levelDim,
    levelOffset(parentLevel),
    parentDim,
    state.cells,
  ];
  uints.forEach((value, index) => view.setUint32(index * 4, value, true));

  const floats = [
    0.0009,
    Number(ui.gravity.value),
    Number(ui.softening.value),
    Number(ui.glow.value),
    state.elapsed,
    ui.canvas.width / Math.max(1, ui.canvas.height),
    state.yaw,
    state.pitch,
    ui.canvas.width,
    ui.canvas.height,
    render ? 0.25 + Number(ui.glow.value) ** 2 * 0.75 : 0.84,
    render ? state.zoom : Number(ui.speed.value),
  ];
  floats.forEach((value, index) => view.setFloat32(32 + index * 4, value, true));
  return data;
}

function updateComputeParams() {
  if (!state.device || !state.computeParams) return;
  for (let level = 0; level <= state.depth; level++) {
    state.device.queue.writeBuffer(
      state.computeParams,
      level * PARAM_STRIDE,
      makeParams(level),
    );
  }
  for (let level = 0; level < state.depth; level++) {
    state.device.queue.writeBuffer(
      state.computeParams,
      (8 + level) * PARAM_STRIDE,
      makeParams(level, true),
    );
  }
}

function updateRenderParams() {
  state.device.queue.writeBuffer(
    state.renderParams,
    0,
    makeParams(state.depth, false, true),
  );
}

function resizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
  const rawWidth = Math.max(1, ui.canvas.clientWidth * dpr);
  const rawHeight = Math.max(1, ui.canvas.clientHeight * dpr);
  // Million-particle point clouds do not benefit from a 4K/Retina swapchain.
  // Keep roughly two million shaded pixels and let CSS upscale the canvas.
  state.resolutionScale = Math.min(1, Math.sqrt(2_000_000 / (rawWidth * rawHeight)));
  const width = Math.max(1, Math.floor(rawWidth * state.resolutionScale));
  const height = Math.max(1, Math.floor(rawHeight * state.resolutionScale));
  if (ui.canvas.width !== width || ui.canvas.height !== height) {
    ui.canvas.width = width;
    ui.canvas.height = height;
    updateComputeParams();
  }
}

async function createGpu() {
  if (!navigator.gpu) {
    throw new Error("This browser does not expose navigator.gpu. Use current Chromium with WebGPU enabled.");
  }
  state.adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!state.adapter) {
    throw new Error("No WebGPU adapter was returned. Check browser GPU diagnostics and hardware acceleration.");
  }

  state.device = await state.adapter.requestDevice({
    label: "Rainbow FMM device",
  });
  state.device.addEventListener("uncapturederror", (event) => {
    showError(`WebGPU validation error: ${event.error.message}`);
  });
  state.device.lost.then((info) => {
    if (info.reason !== "destroyed") {
      showError(`WebGPU device lost: ${info.message || info.reason}`);
    }
  });

  const info = state.adapter.info || {};
  ui.gpuName.textContent = info.device || info.description || info.architecture || "high-performance WebGPU";

  state.context = ui.canvas.getContext("webgpu");
  state.format = navigator.gpu.getPreferredCanvasFormat();
  state.context.configure({
    device: state.device,
    format: state.format,
    alphaMode: "opaque",
  });
}

async function createPipelines() {
  const device = state.device;
  const module = device.createShaderModule({
    label: "Rust-embedded Rainbow FMM WGSL",
    code: state.shader,
  });
  const compilation = await module.getCompilationInfo();
  const failures = compilation.messages.filter((message) => message.type === "error");
  if (failures.length) {
    throw new Error(failures.map((message) => `WGSL ${message.lineNum}:${message.linePos} ${message.message}`).join("\n"));
  }

  state.computeLayout = device.createBindGroupLayout({
    label: "FMM compute resources",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: PARAM_BYTES },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
    ],
  });
  const computePipelineLayout = device.createPipelineLayout({
    label: "FMM compute pipeline layout",
    bindGroupLayouts: [state.computeLayout],
  });

  const entryPoints = [
    "initialize",
    "clear_cells",
    "deposit",
    "upward",
    "m2l",
    "downward",
    "near_field",
    "integrate",
  ];
  const computePipelines = await Promise.all(entryPoints.map(async (entryPoint) => [
    entryPoint,
    await device.createComputePipelineAsync({
      label: `FMM ${entryPoint}`,
      layout: computePipelineLayout,
      compute: { module, entryPoint },
    }),
  ]));
  Object.assign(state.pipelines, Object.fromEntries(computePipelines));

  const emptyLayout = device.createBindGroupLayout({ entries: [] });
  state.renderLayout = device.createBindGroupLayout({
    label: "Particle render resources",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform", minBindingSize: PARAM_BYTES },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "read-only-storage" },
      },
    ],
  });
  const renderPipelineLayout = device.createPipelineLayout({
    label: "Rainbow render pipeline layout",
    bindGroupLayouts: [emptyLayout, state.renderLayout],
  });
  state.pipelines.render = await device.createRenderPipelineAsync({
    label: "Additive rainbow particle renderer",
    layout: renderPipelineLayout,
    vertex: { module, entryPoint: "render_vertex" },
    fragment: {
      module,
      entryPoint: "render_fragment",
      targets: [{
        format: state.format,
        blend: {
          color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
        },
      }],
    },
    primitive: { topology: "triangle-list", cullMode: "none" },
  });
}

function createBuffer(label, size, usage) {
  return state.device.createBuffer({ label, size, usage });
}

function destroySimulationBuffers() {
  if (!state.buffers) return;
  Object.values(state.buffers).forEach((buffer) => buffer.destroy());
  state.buffers = null;
}

async function rebuildSimulation(requestedCount = state.count) {
  if (state.rebuilding) return;
  state.rebuilding = true;
  state.ready = false;
  ui.runState.textContent = "BUILDING SPARSE OCTREE";
  try {
    await state.device.queue.onSubmittedWorkDone();
    destroySimulationBuffers();

    state.count = state.wasm.exports.rainbow_clamp_particle_count(requestedCount);
    state.depth = state.wasm.exports.rainbow_depth_for_particles(state.count);
    state.cells = state.wasm.exports.rainbow_total_cells(state.depth);
    // The force solve still updates every body. Rendering a deterministic
    // quarter-million sample avoids six million blended vertices per frame.
    state.renderStride = Math.max(1, Math.ceil(state.count / 262_144));
    state.renderCount = Math.ceil(state.count / state.renderStride);

    const usage = GPUBufferUsage;
    state.buffers = {
      particles: createBuffer(
        "Particles: position/mass + velocity/hue",
        state.count * PARTICLE_BYTES,
        usage.STORAGE | usage.COPY_DST | usage.COPY_SRC,
      ),
      moments: createBuffer(
        "Sparse octree fixed-point moments",
        state.cells * MOMENT_BYTES,
        usage.STORAGE | usage.COPY_DST,
      ),
      fields: createBuffer(
        "FMM local acceleration and tidal fields",
        state.cells * FIELD_BYTES,
        usage.STORAGE | usage.COPY_DST,
      ),
    };

    state.computeBindGroup = state.device.createBindGroup({
      label: "FMM compute bind group",
      layout: state.computeLayout,
      entries: [
        { binding: 0, resource: { buffer: state.computeParams, size: PARAM_BYTES } },
        { binding: 1, resource: { buffer: state.buffers.particles } },
        { binding: 2, resource: { buffer: state.buffers.moments } },
        { binding: 3, resource: { buffer: state.buffers.fields } },
      ],
    });
    state.renderBindGroup = state.device.createBindGroup({
      label: "Rainbow render bind group",
      layout: state.renderLayout,
      entries: [
        { binding: 0, resource: { buffer: state.renderParams } },
        { binding: 1, resource: { buffer: state.buffers.particles } },
      ],
    });

    updateComputeParams();
    const encoder = state.device.createCommandEncoder({ label: "Seed particle system" });
    const pass = encoder.beginComputePass();
    pass.setPipeline(state.pipelines.initialize);
    pass.setBindGroup(0, state.computeBindGroup, [state.depth * PARAM_STRIDE]);
    pass.dispatchWorkgroups(Math.ceil(state.count / 256));
    pass.end();
    state.device.queue.submit([encoder.finish()]);
    await state.device.queue.onSubmittedWorkDone();

    ui.particleStat.textContent = state.count.toLocaleString("en-US");
    ui.treeStat.textContent = `${state.cells.toLocaleString("en-US")} cells · ${state.renderCount.toLocaleString("en-US")} visible`;
    ui.runState.textContent = state.running ? "SIMULATION LIVE" : "SIMULATION PAUSED";
    state.ready = true;
  } finally {
    state.rebuilding = false;
  }
}

function encodeSolver(encoder) {
  const pass = encoder.beginComputePass({ label: "Linear FMM solve" });

  pass.setPipeline(state.pipelines.clear_cells);
  pass.setBindGroup(0, state.computeBindGroup, [state.depth * PARAM_STRIDE]);
  pass.dispatchWorkgroups(Math.ceil(state.cells / 256));

  pass.setPipeline(state.pipelines.deposit);
  pass.dispatchWorkgroups(Math.ceil(state.count / 256));

  pass.setPipeline(state.pipelines.upward);
  for (let level = state.depth - 1; level >= 0; level--) {
    pass.setBindGroup(0, state.computeBindGroup, [(8 + level) * PARAM_STRIDE]);
    pass.dispatchWorkgroups(Math.ceil(levelCells(level) / 64));
  }

  for (let level = 1; level <= state.depth; level++) {
    const offset = level * PARAM_STRIDE;
    pass.setBindGroup(0, state.computeBindGroup, [offset]);
    pass.setPipeline(state.pipelines.m2l);
    pass.dispatchWorkgroups(Math.ceil(levelCells(level) / 64));
    pass.setPipeline(state.pipelines.downward);
    pass.dispatchWorkgroups(Math.ceil(levelCells(level) / 64));
  }

  pass.setPipeline(state.pipelines.near_field);
  for (let level = 1; level <= state.depth; level++) {
    pass.setBindGroup(0, state.computeBindGroup, [level * PARAM_STRIDE]);
    pass.dispatchWorkgroups(Math.ceil(levelCells(level) / 64));
  }
  pass.setBindGroup(0, state.computeBindGroup, [state.depth * PARAM_STRIDE]);
  pass.setPipeline(state.pipelines.integrate);
  pass.dispatchWorkgroups(Math.ceil(state.count / 256));
  pass.end();
}

function encodeRender(encoder) {
  updateRenderParams();
  const pass = encoder.beginRenderPass({
    label: "Rainbow particle field",
    colorAttachments: [{
      view: state.context.getCurrentTexture().createView(),
      clearValue: { r: 0.012, g: 0.014, b: 0.019, a: 1 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  pass.setPipeline(state.pipelines.render);
  pass.setBindGroup(1, state.renderBindGroup);
  pass.draw(6, state.renderCount);
  pass.end();
}

function frame(now) {
  if (!state.device) return;
  try {
    resizeCanvas();
    const frameInterval = Math.min(100, now - state.lastFrameAt);
    state.lastFrameAt = now;
    state.elapsed += frameInterval * 0.001;

    if (state.ready) {
      const encoder = state.device.createCommandEncoder({ label: "Rainbow FMM frame" });
      if (state.running) encodeSolver(encoder);
      encodeRender(encoder);
      state.device.queue.submit([encoder.finish()]);
      state.submittedFrames += 1;

      // Do not await queue.onSubmittedWorkDone() here. It is a full GPU fence
      // and Firefox can couple it to swapchain presentation, leaving the GPU
      // idle between frames. The queue already preserves command order.
      state.frameSamples.push(frameInterval);
      if (state.frameSamples.length > 24) state.frameSamples.shift();
      const average = state.frameSamples.reduce((sum, value) => sum + value, 0) / state.frameSamples.length;
      ui.frameStat.textContent = `${average.toFixed(1)} ms`;
      ui.fpsStat.textContent = `${Math.min(999, 1000 / Math.max(average, 0.1)).toFixed(0)}`;
    }
    requestAnimationFrame(frame);
  } catch (error) {
    showError(error);
  }
}

async function runOneSolverStep() {
  if (!state.ready || state.rebuilding) {
    throw new Error("Simulation is not ready for a diagnostic step.");
  }
  const wasReady = state.ready;
  state.ready = false;
  try {
    await state.device.queue.onSubmittedWorkDone();
    const started = performance.now();
    const encoder = state.device.createCommandEncoder({ label: "Single diagnostic FMM step" });
    encodeSolver(encoder);
    state.device.queue.submit([encoder.finish()]);
    await state.device.queue.onSubmittedWorkDone();
    return performance.now() - started;
  } finally {
    state.ready = wasReady;
  }
}

function bindControls() {
  const syncOutputs = () => {
    ui.gravityValue.textContent = Number(ui.gravity.value).toFixed(2);
    ui.softeningValue.textContent = Number(ui.softening.value).toFixed(3);
    ui.speedValue.textContent = `${Number(ui.speed.value).toFixed(2)}×`;
    ui.glowValue.textContent = Number(ui.glow.value).toFixed(2);
    updateComputeParams();
  };
  [ui.gravity, ui.softening, ui.speed, ui.glow].forEach((input) => {
    input.addEventListener("input", syncOutputs);
  });

  ui.count.addEventListener("change", () => rebuildSimulation(Number(ui.count.value)).catch(showError));
  ui.reseed.addEventListener("click", () => rebuildSimulation(state.count).catch(showError));
  ui.toggle.addEventListener("click", () => {
    state.running = !state.running;
    ui.toggle.classList.toggle("paused", !state.running);
    ui.toggle.querySelector("span:last-child").textContent = state.running ? "Pause" : "Resume";
    ui.runState.textContent = state.running ? "SIMULATION LIVE" : "SIMULATION PAUSED";
  });
  ui.collapse.addEventListener("click", () => {
    const collapsed = ui.deck.classList.toggle("collapsed");
    ui.collapse.textContent = collapsed ? "+" : "−";
    ui.collapse.setAttribute("aria-expanded", String(!collapsed));
  });
  ui.fullscreen.addEventListener("click", () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen();
  });

  ui.canvas.addEventListener("pointerdown", (event) => {
    state.dragging = true;
    state.pointerX = event.clientX;
    state.pointerY = event.clientY;
    ui.canvas.setPointerCapture(event.pointerId);
  });
  ui.canvas.addEventListener("pointermove", (event) => {
    if (!state.dragging) return;
    state.yaw += (event.clientX - state.pointerX) * 0.005;
    state.pitch = Math.max(-1.1, Math.min(1.1, state.pitch + (event.clientY - state.pointerY) * 0.004));
    state.pointerX = event.clientX;
    state.pointerY = event.clientY;
  });
  ui.canvas.addEventListener("pointerup", () => { state.dragging = false; });
  ui.canvas.addEventListener("pointercancel", () => { state.dragging = false; });
  ui.canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    state.zoom = Math.max(1.55, Math.min(3.65, state.zoom + event.deltaY * 0.0014));
  }, { passive: false });

  window.addEventListener("resize", resizeCanvas);
  syncOutputs();
}

function showError(error) {
  const message = error instanceof Error ? error.message : String(error);
  console.error(error);
  ui.errorMessage.textContent = message;
  ui.errorPanel.hidden = false;
  ui.runState.textContent = "ENGINE ERROR";
}

async function main() {
  try {
    const query = new URLSearchParams(location.search);
    if (query.has("particles")) {
      const requested = query.get("particles");
      if (![...ui.count.options].some((option) => option.value === requested)) {
        ui.count.add(new Option(`${Number(requested).toLocaleString("en-US")} · diagnostic`, requested));
      }
      ui.count.value = requested;
    }
    if (query.get("paused") === "1") {
      state.running = false;
      ui.toggle.classList.add("paused");
      ui.toggle.querySelector("span:last-child").textContent = "Resume";
    }
    state.count = Number(ui.count.value);
    state.wasm = await instantiateWasm("./pkg/rainbow_fmm.wasm?v=visibility-perf-2");
    if (state.wasm.exports.rainbow_api_version() !== 1) {
      throw new Error("Rust/Wasm API version mismatch.");
    }
    state.shader = wasmString(
      state.wasm.exports.rainbow_shader_ptr(),
      state.wasm.exports.rainbow_shader_len(),
    );
    state.algorithm = wasmString(
      state.wasm.exports.rainbow_algorithm_ptr(),
      state.wasm.exports.rainbow_algorithm_len(),
    );
    console.info(`Rainbow FMM: ${state.algorithm}`);

    bindControls();
    resizeCanvas();
    await createGpu();

    state.computeParams = createBuffer(
      "Rust FMM pass plans",
      PARAM_SLOTS * PARAM_STRIDE,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
    state.renderParams = createBuffer(
      "Camera and render parameters",
      PARAM_BYTES,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );

    await createPipelines();
    await rebuildSimulation(state.count);
    state.runOneSolverStep = runOneSolverStep;
    window.__rainbowDiagnostics = state;
    state.lastFrameAt = performance.now();
    requestAnimationFrame(frame);
  } catch (error) {
    showError(error);
  }
}

main();
