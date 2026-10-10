const $ = (selector) => document.querySelector(selector);
const TOOL_MODES = Object.freeze({
  orbit: 0,
  pull: 1,
  push: 2,
  vortex: 3,
});

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
  controlStrength: $("#control-strength"),
  controlStrengthValue: $("#control-strength-value"),
  controlRadius: $("#control-radius"),
  controlRadiusValue: $("#control-radius-value"),
  toolButtons: [...document.querySelectorAll(".tool-button")],
  shepherdCursor: $("#shepherd-cursor"),
  interactionHint: $("#interaction-hint"),
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

const BASE_DT = 0.0009;
const REFERENCE_FRAME_MS = 1000 / 60;

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
  accumulationBindGroup: null,
  blurBindGroup: null,
  computeParams: null,
  renderParams: null,
  accumulationTexture: null,
  accumulationView: null,
  blurTexture: null,
  blurView: null,
  count: 1 << 20,
  depth: 6,
  cells: 299_593,
  renderCount: 1 << 20,
  renderStride: 1,
  resolutionScale: 1,
  running: true,
  ready: false,
  rebuilding: false,
  yaw: -0.18,
  pitch: -0.14,
  zoom: 1.78,
  tool: "orbit",
  controlActive: false,
  controlPoint: [0, 0, 0],
  controlAxis: [0, 0, 1],
  pointerInside: false,
  cursorX: 0,
  cursorY: 0,
  activePointerId: null,
  dragging: false,
  pointerX: 0,
  pointerY: 0,
  elapsed: 0,
  frameSamples: [],
  frameDt: BASE_DT,
  lastFrameAt: performance.now(),
  submittedFrames: 0,
};

const PARAM_BYTES = 128;
const PARAM_STRIDE = 256;
const PARAM_SLOTS = 16;
const PARTICLE_BYTES = 32;
const MOMENT_BYTES = 16;
const FIELD_BYTES = 48;
const ACCUMULATION_FORMAT = "rgba16float";
const ACCUMULATION_SCALE = 0.5;

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
    render ? Math.cos(state.yaw) : state.frameDt,
    render ? Math.sin(state.yaw) : Number(ui.gravity.value),
    render ? Math.cos(state.pitch) : Number(ui.softening.value),
    Number(ui.glow.value),
    state.elapsed,
    ui.canvas.width / Math.max(1, ui.canvas.height),
    state.yaw,
    state.pitch,
    ui.canvas.width,
    ui.canvas.height,
    render ? 1.0 : 0.84,
    render ? state.zoom : Number(ui.speed.value),
    state.controlPoint[0],
    state.controlPoint[1],
    state.controlPoint[2],
    state.controlActive && state.tool !== "orbit" ? 1 : 0,
    state.controlAxis[0],
    state.controlAxis[1],
    state.controlAxis[2],
    TOOL_MODES[state.tool] || 0,
    Number(ui.controlStrength.value),
    Number(ui.controlRadius.value),
    render ? Math.sin(state.pitch) : 0,
    0,
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

function rebuildAccumulationTarget() {
  if (!state.device || !state.resolveLayout || !ui.canvas.width || !ui.canvas.height) return;
  state.accumulationTexture?.destroy();
  state.blurTexture?.destroy();
  const width = Math.max(1, Math.ceil(ui.canvas.width * ACCUMULATION_SCALE));
  const height = Math.max(1, Math.ceil(ui.canvas.height * ACCUMULATION_SCALE));
  state.accumulationTexture = state.device.createTexture({
    label: "HDR particle color and density",
    size: [width, height],
    format: ACCUMULATION_FORMAT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  state.accumulationView = state.accumulationTexture.createView();
  state.blurTexture = state.device.createTexture({
    label: "Horizontally reconstructed HDR field",
    size: [width, height],
    format: ACCUMULATION_FORMAT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  state.blurView = state.blurTexture.createView();
  state.accumulationBindGroup = state.device.createBindGroup({
    label: "Raw HDR field bind group",
    layout: state.resolveLayout,
    entries: [
      { binding: 0, resource: state.accumulationView },
      { binding: 1, resource: state.resolveSampler },
    ],
  });
  state.blurBindGroup = state.device.createBindGroup({
    label: "Horizontally reconstructed field bind group",
    layout: state.resolveLayout,
    entries: [
      { binding: 0, resource: state.blurView },
      { binding: 1, resource: state.resolveSampler },
    ],
  });
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
    rebuildAccumulationTarget();
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

  // The tree-reading passes (M2L, downward, near field) never touch atomics,
  // so they bind the moments buffer read-only and fetch a source cell with one
  // vector load instead of four atomic loads.
  state.treeLayout = device.createBindGroupLayout({
    label: "FMM read-only tree resources",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: PARAM_BYTES },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
    ],
  });
  const emptyGroupLayout = device.createBindGroupLayout({ entries: [] });
  const treePipelineLayout = device.createPipelineLayout({
    label: "FMM tree pipeline layout",
    bindGroupLayouts: [
      emptyGroupLayout,
      emptyGroupLayout,
      emptyGroupLayout,
      state.treeLayout,
    ],
  });

  const entryPoints = [
    "initialize",
    "clear_cells",
    "deposit",
    "upward",
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
  const treePipelines = await Promise.all(["m2l", "downward", "near_field"].map(async (entryPoint) => [
    entryPoint,
    await device.createComputePipelineAsync({
      label: `FMM ${entryPoint}`,
      layout: treePipelineLayout,
      compute: { module, entryPoint },
    }),
  ]));
  Object.assign(state.pipelines, Object.fromEntries(computePipelines), Object.fromEntries(treePipelines));

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
    label: "Rainbow HDR accumulation pipeline layout",
    bindGroupLayouts: [emptyLayout, state.renderLayout],
  });
  state.pipelines.render = await device.createRenderPipelineAsync({
    label: "HDR rainbow particle accumulation",
    layout: renderPipelineLayout,
    vertex: { module, entryPoint: "render_vertex" },
    fragment: {
      module,
      entryPoint: "render_fragment",
      targets: [{
        format: ACCUMULATION_FORMAT,
        blend: {
          color: { srcFactor: "one", dstFactor: "one", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
        },
      }],
    },
    primitive: { topology: "point-list", cullMode: "none" },
  });

  state.resolveLayout = device.createBindGroupLayout({
    label: "HDR density resolve texture",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: "float" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: { type: "filtering" },
      },
    ],
  });
  state.resolveSampler = device.createSampler({
    label: "Smooth HDR field reconstruction",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
    minFilter: "linear",
    magFilter: "linear",
  });
  const resolvePipelineLayout = device.createPipelineLayout({
    label: "HDR field reconstruction pipeline layout",
    bindGroupLayouts: [emptyLayout, state.renderLayout, state.resolveLayout],
  });
  state.pipelines.blur = await device.createRenderPipelineAsync({
    label: "Horizontal HDR field reconstruction",
    layout: resolvePipelineLayout,
    vertex: { module, entryPoint: "resolve_vertex" },
    fragment: {
      module,
      entryPoint: "blur_horizontal_fragment",
      targets: [{ format: ACCUMULATION_FORMAT }],
    },
    primitive: { topology: "triangle-list", cullMode: "none" },
  });
  state.pipelines.blurVertical = await device.createRenderPipelineAsync({
    label: "Vertical HDR field reconstruction",
    layout: resolvePipelineLayout,
    vertex: { module, entryPoint: "resolve_vertex" },
    fragment: {
      module,
      entryPoint: "blur_vertical_fragment",
      targets: [{ format: ACCUMULATION_FORMAT }],
    },
    primitive: { topology: "triangle-list", cullMode: "none" },
  });
  state.pipelines.resolve = await device.createRenderPipelineAsync({
    label: "Density-normalized rainbow resolve",
    layout: resolvePipelineLayout,
    vertex: { module, entryPoint: "resolve_vertex" },
    fragment: {
      module,
      entryPoint: "resolve_fragment",
      targets: [{ format: state.format }],
    },
    primitive: { topology: "triangle-list", cullMode: "none" },
  });
  rebuildAccumulationTarget();
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
    // Every simulated body contributes to the compact HDR density field.
    state.renderStride = 1;
    state.renderCount = state.count;

    const usage = GPUBufferUsage;
    state.buffers = {
      particles: createBuffer(
        "Particles: position/dynamical hue + velocity/excitation",
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
    state.treeBindGroup = state.device.createBindGroup({
      label: "FMM read-only tree bind group",
      layout: state.treeLayout,
      entries: [
        { binding: 0, resource: { buffer: state.computeParams, size: PARAM_BYTES } },
        { binding: 1, resource: { buffer: state.buffers.moments } },
        { binding: 2, resource: { buffer: state.buffers.fields } },
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
    refreshRunState(true);
    state.ready = true;
  } finally {
    state.rebuilding = false;
  }
}

function encodeSolver(encoder) {
  updateComputeParams();
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
    pass.setBindGroup(3, state.treeBindGroup, [offset]);
    pass.setPipeline(state.pipelines.m2l);
    pass.dispatchWorkgroups(Math.ceil(levelCells(level) / 64));
    pass.setPipeline(state.pipelines.downward);
    pass.dispatchWorkgroups(Math.ceil(levelCells(level) / 64));
  }

  pass.setPipeline(state.pipelines.near_field);
  for (let level = 1; level <= state.depth; level++) {
    pass.setBindGroup(3, state.treeBindGroup, [level * PARAM_STRIDE]);
    pass.dispatchWorkgroups(Math.ceil(levelCells(level) / 64));
  }
  pass.setBindGroup(0, state.computeBindGroup, [state.depth * PARAM_STRIDE]);
  pass.setPipeline(state.pipelines.integrate);
  pass.dispatchWorkgroups(Math.ceil(state.count / 256));
  pass.end();
}

function encodeRender(encoder, targetView = null) {
  updateRenderParams();
  const accumulationPass = encoder.beginRenderPass({
    label: "HDR particle color and density accumulation",
    colorAttachments: [{
      view: state.accumulationView,
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  accumulationPass.setPipeline(state.pipelines.render);
  accumulationPass.setBindGroup(1, state.renderBindGroup);
  accumulationPass.draw(1, state.renderCount);
  accumulationPass.end();

  const blurPass = encoder.beginRenderPass({
    label: "Horizontal HDR field reconstruction",
    colorAttachments: [{
      view: state.blurView,
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  blurPass.setPipeline(state.pipelines.blur);
  blurPass.setBindGroup(1, state.renderBindGroup);
  blurPass.setBindGroup(2, state.accumulationBindGroup);
  blurPass.draw(3);
  blurPass.end();

  const verticalBlurPass = encoder.beginRenderPass({
    label: "Vertical HDR field reconstruction",
    colorAttachments: [{
      view: state.accumulationView,
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  verticalBlurPass.setPipeline(state.pipelines.blurVertical);
  verticalBlurPass.setBindGroup(1, state.renderBindGroup);
  verticalBlurPass.setBindGroup(2, state.blurBindGroup);
  verticalBlurPass.draw(3);
  verticalBlurPass.end();

  const resolvePass = encoder.beginRenderPass({
    label: "Density-normalized rainbow resolve",
    colorAttachments: [{
      view: targetView || state.context.getCurrentTexture().createView(),
      clearValue: { r: 0.008, g: 0.010, b: 0.016, a: 1 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  resolvePass.setPipeline(state.pipelines.resolve);
  resolvePass.setBindGroup(1, state.renderBindGroup);
  resolvePass.setBindGroup(2, state.accumulationBindGroup);
  resolvePass.draw(3);
  resolvePass.end();
}

function frame(now) {
  if (!state.device) return;
  try {
    resizeCanvas();
    const frameInterval = Math.min(100, now - state.lastFrameAt);
    state.lastFrameAt = now;
    state.elapsed += frameInterval * 0.001;
    state.frameDt = BASE_DT * (frameInterval / REFERENCE_FRAME_MS);

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

async function captureRenderFrame() {
  if (!state.ready || state.rebuilding) {
    throw new Error("Simulation is not ready for a render capture.");
  }
  const wasReady = state.ready;
  state.ready = false;
  let texture;
  let readback;
  try {
    await state.device.queue.onSubmittedWorkDone();
    const width = ui.canvas.width;
    const height = ui.canvas.height;
    const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
    texture = state.device.createTexture({
      label: "Diagnostic render target",
      size: [width, height],
      format: state.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    readback = state.device.createBuffer({
      label: "Diagnostic render readback",
      size: bytesPerRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = state.device.createCommandEncoder({
      label: "Diagnostic render capture",
    });
    encodeRender(encoder, texture.createView());
    encoder.copyTextureToBuffer(
      { texture },
      { buffer: readback, bytesPerRow, rowsPerImage: height },
      [width, height],
    );
    state.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);

    const source = new Uint8Array(readback.getMappedRange());
    const rgba = new Uint8ClampedArray(width * height * 4);
    const bgra = state.format.startsWith("bgra");
    for (let y = 0; y < height; y += 1) {
      const sourceRow = y * bytesPerRow;
      const targetRow = y * width * 4;
      for (let x = 0; x < width; x += 1) {
        const sourcePixel = sourceRow + x * 4;
        const targetPixel = targetRow + x * 4;
        rgba[targetPixel] = source[sourcePixel + (bgra ? 2 : 0)];
        rgba[targetPixel + 1] = source[sourcePixel + 1];
        rgba[targetPixel + 2] = source[sourcePixel + (bgra ? 0 : 2)];
        rgba[targetPixel + 3] = source[sourcePixel + 3];
      }
    }

    document.querySelector("#gpu-readback")?.remove();
    const capture = document.createElement("canvas");
    capture.id = "gpu-readback";
    capture.width = width;
    capture.height = height;
    capture.style.cssText = [
      "position:fixed",
      "inset:0",
      "z-index:1000",
      "max-width:100vw",
      "max-height:100vh",
    ].join(";");
    capture.getContext("2d").putImageData(
      new ImageData(rgba, width, height),
      0,
      0,
    );
    document.body.append(capture);
    return { width, height, selector: "#gpu-readback" };
  } finally {
    if (readback?.mapState === "mapped") readback.unmap();
    readback?.destroy();
    texture?.destroy();
    state.ready = wasReady;
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

function inverseRotateCamera(point) {
  const [x, y, z] = point;
  const cp = Math.cos(state.pitch);
  const sp = Math.sin(state.pitch);
  const cy = Math.cos(state.yaw);
  const sy = Math.sin(state.yaw);
  const unpitchedY = cp * y + sp * z;
  const unpitchedZ = -sp * y + cp * z;
  return [
    cy * x - sy * unpitchedZ,
    unpitchedY,
    sy * x + cy * unpitchedZ,
  ];
}

function refreshRunState(force = false) {
  if (state.rebuilding && !force) return;
  if (!state.running) {
    ui.runState.textContent = "SIMULATION PAUSED";
  } else if (state.controlActive) {
    ui.runState.textContent = `${state.tool.toUpperCase()} FIELD · EXTERNAL WORK`;
  } else {
    ui.runState.textContent = "SIMULATION LIVE";
  }
}

function updateControlCursorSize() {
  const diameter = Number(ui.controlRadius.value) *
    1.92 / state.zoom * ui.canvas.clientHeight;
  const size = Math.max(36, diameter);
  ui.shepherdCursor.style.width = `${size}px`;
  ui.shepherdCursor.style.height = `${size}px`;
}

function updateControlFromPointer(event) {
  const rect = ui.canvas.getBoundingClientRect();
  const shellRect = ui.canvas.parentElement.getBoundingClientRect();
  const x = Math.max(0, Math.min(rect.width, event.clientX - rect.left));
  const y = Math.max(0, Math.min(rect.height, event.clientY - rect.top));
  state.cursorX = event.clientX - shellRect.left;
  state.cursorY = event.clientY - shellRect.top;
  ui.shepherdCursor.style.left = `${state.cursorX}px`;
  ui.shepherdCursor.style.top = `${state.cursorY}px`;

  const ndcX = x / Math.max(rect.width, 1) * 2 - 1;
  const ndcY = 1 - y / Math.max(rect.height, 1) * 2;
  const aspect = rect.width / Math.max(rect.height, 1);
  const cameraPoint = [
    ndcX * aspect * state.zoom / 1.92,
    ndcY * state.zoom / 1.92,
    0,
  ];
  state.controlPoint = inverseRotateCamera(cameraPoint)
    .map((value) => Math.max(-0.95, Math.min(0.95, value)));
  state.controlAxis = inverseRotateCamera([0, 0, 1]);
}

function setTool(tool) {
  if (!(tool in TOOL_MODES)) return;
  state.tool = tool;
  state.controlActive = false;
  state.dragging = false;
  ui.toolButtons.forEach((button) => {
    const active = button.dataset.tool === tool;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  ui.shepherdCursor.dataset.tool = tool;
  ui.shepherdCursor.classList.remove("active");
  ui.shepherdCursor.hidden = tool === "orbit" || !state.pointerInside;
  const hints = {
    orbit: "Drag to orbit · Wheel to zoom · Keys 1–4 select tools",
    pull: "Hold and drag to attract · External work is explicit · Key 1 returns to orbit",
    push: "Hold and drag to repel · External work is explicit · Key 1 returns to orbit",
    vortex: "Hold and drag to spin · Adds angular momentum · Key 1 returns to orbit",
  };
  ui.interactionHint.textContent = hints[tool];
  updateComputeParams();
  refreshRunState();
}

function finishPointerInteraction(event) {
  const wasControlling = state.controlActive;
  state.dragging = false;
  state.controlActive = false;
  ui.shepherdCursor.classList.remove("active");
  ui.shepherdCursor.hidden =
    state.tool === "orbit" || !state.pointerInside;
  if (
    event &&
    state.activePointerId !== null &&
    ui.canvas.hasPointerCapture(state.activePointerId)
  ) {
    ui.canvas.releasePointerCapture(state.activePointerId);
  }
  state.activePointerId = null;
  if (wasControlling) updateComputeParams();
  refreshRunState();
}

function bindControls() {
  const syncOutputs = () => {
    ui.gravityValue.textContent = Number(ui.gravity.value).toFixed(2);
    ui.softeningValue.textContent = Number(ui.softening.value).toFixed(3);
    ui.speedValue.textContent = `${Number(ui.speed.value).toFixed(2)}×`;
    ui.glowValue.textContent = `${Number(ui.glow.value).toFixed(1)} px`;
    ui.controlStrengthValue.textContent =
      Number(ui.controlStrength.value).toFixed(2);
    ui.controlRadiusValue.textContent =
      Number(ui.controlRadius.value).toFixed(2);
    updateControlCursorSize();
    updateComputeParams();
  };
  [
    ui.gravity,
    ui.softening,
    ui.speed,
    ui.glow,
    ui.controlStrength,
    ui.controlRadius,
  ].forEach((input) => {
    input.addEventListener("input", syncOutputs);
  });

  ui.toolButtons.forEach((button) => {
    button.addEventListener("click", () => setTool(button.dataset.tool));
  });
  ui.count.addEventListener("change", () => rebuildSimulation(Number(ui.count.value)).catch(showError));
  ui.reseed.addEventListener("click", () => rebuildSimulation(state.count).catch(showError));
  ui.toggle.addEventListener("click", () => {
    state.running = !state.running;
    ui.toggle.classList.toggle("paused", !state.running);
    ui.toggle.querySelector("span:last-child").textContent = state.running ? "Pause" : "Resume";
    refreshRunState();
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
    if (event.button !== 0) return;
    event.preventDefault();
    state.pointerInside = true;
    state.activePointerId = event.pointerId;
    state.pointerX = event.clientX;
    state.pointerY = event.clientY;
    updateControlFromPointer(event);
    ui.canvas.setPointerCapture(event.pointerId);
    if (state.tool === "orbit") {
      state.dragging = true;
    } else {
      state.controlActive = true;
      ui.shepherdCursor.hidden = false;
      ui.shepherdCursor.classList.add("active");
      updateComputeParams();
      refreshRunState();
    }
  });
  ui.canvas.addEventListener("pointermove", (event) => {
    if (state.tool !== "orbit") {
      updateControlFromPointer(event);
      ui.shepherdCursor.hidden = false;
      if (state.controlActive) updateComputeParams();
    } else if (state.dragging) {
      state.yaw += (event.clientX - state.pointerX) * 0.005;
      state.pitch = Math.max(
        -1.1,
        Math.min(
          1.1,
          state.pitch + (event.clientY - state.pointerY) * 0.004,
        ),
      );
      state.pointerX = event.clientX;
      state.pointerY = event.clientY;
    }
  });
  ui.canvas.addEventListener("pointerenter", (event) => {
    state.pointerInside = true;
    if (state.tool !== "orbit") {
      updateControlFromPointer(event);
      ui.shepherdCursor.hidden = false;
    }
  });
  ui.canvas.addEventListener("pointerleave", () => {
    state.pointerInside = false;
    if (!state.controlActive) ui.shepherdCursor.hidden = true;
  });
  ui.canvas.addEventListener("pointerup", finishPointerInteraction);
  ui.canvas.addEventListener("pointercancel", finishPointerInteraction);
  ui.canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    state.zoom = Math.max(1.55, Math.min(3.65, state.zoom + event.deltaY * 0.0014));
    updateControlCursorSize();
  }, { passive: false });

  window.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLSelectElement) {
      return;
    }
    const tool = {
      Digit1: "orbit",
      Digit2: "pull",
      Digit3: "push",
      Digit4: "vortex",
    }[event.code];
    if (tool) setTool(tool);
  });
  window.addEventListener("blur", () => finishPointerInteraction());
  window.addEventListener("resize", () => {
    resizeCanvas();
    updateControlCursorSize();
  });
  syncOutputs();
  setTool("orbit");
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
    state.wasm = await instantiateWasm("./pkg/rainbow_fmm.wasm?v=wide-world-20");
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
    state.captureRenderFrame = captureRenderFrame;
    window.__rainbowDiagnostics = state;
    state.lastFrameAt = performance.now();
    requestAnimationFrame(frame);
  } catch (error) {
    showError(error);
  }
}

main();
