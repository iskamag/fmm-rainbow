// Headless native WebGPU harness for the Rainbow FMM shader.
//
// Reuses the exact WGSL payload and the Rust planner from the demo crate, so
// the pass structure and uniform plans match the browser. Runs directly on the
// GPU through wgpu/Vulkan with no display, and uses GPU timestamp queries for
// true per-phase GPU time.
//
//   cargo run --release -- <particles> [steps]
//
// It prints the adapter, the chosen plan, a content digest of the field and
// moment buffers (for cross-checking against the browser), the per-phase GPU
// time of one frame, the amortized wall-clock time of a full solve, and a
// finiteness check over a particle slice.

use std::borrow::Cow;
use std::collections::HashMap;

const PARAM_BYTES: u64 = 128;
const PARAM_STRIDE: u64 = 256;
const PARAM_SLOTS: u64 = 16;
const PARTICLE_BYTES: u64 = 32;
const MOMENT_BYTES: u64 = 16;
const FIELD_BYTES: u64 = 48;
// One 2D dispatch row: 65535 workgroups of 256 threads.
const PARTICLE_ROW: u32 = 65535 * 256;

struct Plans {
    level: Vec<Vec<u8>>,  // per level 0..=depth
    upward: Vec<Vec<u8>>, // per parent level 0..depth
    depth: u32,
    cells: u32,
    count: u32,
    scale: f32,
}

/// Serializes one uniform `Params` block. The field order must match
/// `struct Params` in rainbow_fmm.wgsl: eight u32s then twenty-four f32s.
#[allow(clippy::too_many_arguments)]
fn write_params(
    count: u32,
    depth: u32,
    level: u32,
    level_offset: u32,
    level_dim: u32,
    parent_offset: u32,
    parent_dim: u32,
    cells: u32,
    scale: f32,
) -> Vec<u8> {
    let mut data = vec![0u8; PARAM_BYTES as usize];
    let uints: [u32; 8] = [
        count, depth, level, level_offset, level_dim, parent_offset, parent_dim, cells,
    ];
    for (i, v) in uints.iter().enumerate() {
        data[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    let floats: [f32; 24] = [
        0.0009, // dt
        0.34,   // gravity
        0.012,  // softening
        5.0,    // point_size
        0.0,    // time
        1.0,    // aspect
        0.0,    // camera_yaw
        0.0,    // camera_pitch
        0.0,    // viewport_width
        0.0,    // viewport_height
        0.84,   // exposure
        1.0,    // speed
        0.0,    // control_point x
        0.0,    // control_point y
        0.0,    // control_point z
        0.0,    // control_point w
        0.0,    // control axis x
        0.0,    // control axis y
        0.0,    // control axis z
        0.0,    // control axis mode
        0.7,    // control strength
        0.22,   // control radius
        scale,  // fixed-point deposit scale (control_settings.z)
        0.0,
    ];
    for (i, v) in floats.iter().enumerate() {
        data[32 + i * 4..32 + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    data
}

fn build_plans(count: u32) -> Plans {
    let depth = rainbow_fmm::depth_for_particles(count);
    let cells = rainbow_fmm::total_cells(depth);
    let scale = rainbow_fmm::fixed_point_scale(count);
    // level_dim / parent_dim are the cube SIDE (2^level), matching cell_coord,
    // cell_center and linear_index in the shader. `level_cells` is 2^(3*level),
    // the number of cells, and is used only for the dispatch grid.
    let side = |l: u32| 1u32 << l;
    let level: Vec<Vec<u8>> = (0..=depth)
        .map(|l| {
            write_params(
                count,
                depth,
                l,
                rainbow_fmm::level_offset(l),
                side(l),
                rainbow_fmm::level_offset(l.saturating_sub(1)),
                side(l.saturating_sub(1)),
                cells,
                scale,
            )
        })
        .collect();
    let upward: Vec<Vec<u8>> = (0..depth)
        .map(|l| {
            write_params(
                count,
                depth,
                l,
                rainbow_fmm::level_offset(l),
                side(l),
                rainbow_fmm::level_offset(l + 1),
                side(l + 1),
                cells,
                scale,
            )
        })
        .collect();
    Plans { level, upward, depth, cells, count, scale }
}

/// Reads a device buffer into a `Vec<u32>` (the source must be 4-byte aligned).
fn read_u32(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, len: u64) -> Vec<u32> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(src, 0, &readback, 0, len);
    queue.submit([enc.finish()]);
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
        .unwrap();
    let data = slice.get_mapped_range().unwrap();
    let words: &[u32] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u32, (len / 4) as usize) };
    words.to_vec()
}

/// FNV-1a content digest of a device buffer, for cross-checking runs.
fn digest(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, len: u64) -> u32 {
    let words = read_u32(device, queue, src, len);
    let mut h: u32 = 2166136261;
    for w in words {
        for b in w.to_le_bytes() {
            h ^= b as u32;
            h = h.wrapping_mul(16777619);
        }
    }
    h
}

fn main() {
    let count: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1 << 20);
    let repeat: u32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(40);

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .expect("no adapter");

    let info = adapter.get_info();
    println!("adapter: {} ({:?}, {:?})", info.name, info.backend, info.device_type);
    if !adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
        println!("(timestamp-query unavailable; per-phase times will be absent)");
    }

    // The multi-million budgets exceed the default 128 MiB storage-binding cap,
    // so request the adapter's own maxima (as the browser does).
    let limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("rainbow-native"),
        required_features: wgpu::Features::TIMESTAMP_QUERY,
        required_limits: limits,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::default(),
    }))
    .expect("no device");

    let plans = build_plans(count);
    println!(
        "count={} depth={} cells={} scale={}",
        plans.count, plans.depth, plans.cells, plans.scale
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rainbow"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(rainbow_fmm::SHADER)),
    });

    // --- Bind group layouts, mirroring app.js ---
    let storage = |read_only: bool| wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let uniform = wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: true,
        min_binding_size: wgpu::BufferSize::new(PARAM_BYTES),
    };
    let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    };
    let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("compute"),
        entries: &[
            entry(0, uniform.clone()),
            entry(1, storage(false)),
            entry(2, storage(false)),
            entry(3, storage(false)),
            entry(4, storage(false)),
        ],
    });
    let empty_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("empty"),
        entries: &[],
    });
    // Tree-reading passes see the moments buffer read-only.
    let tree_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tree"),
        entries: &[
            entry(0, uniform.clone()),
            entry(1, storage(true)),
            entry(2, storage(false)),
            entry(3, storage(false)),
        ],
    });

    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("compute"),
        bind_group_layouts: &[Some(&compute_layout)],
        immediate_size: 0,
    });
    let tree_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("tree"),
        bind_group_layouts: &[
            Some(&empty_layout),
            Some(&empty_layout),
            Some(&empty_layout),
            Some(&tree_layout),
        ],
        immediate_size: 0,
    });

    let mk = |name: &str, layout: &wgpu::PipelineLayout| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: Some(layout),
            module: &shader,
            entry_point: Some(name),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let mut pipes: HashMap<&str, wgpu::ComputePipeline> = HashMap::new();
    for name in ["initialize", "clear_cells", "deposit", "upward"] {
        pipes.insert(name, mk(name, &compute_pipeline_layout));
    }
    for name in ["m2l", "downward", "near_field", "integrate"] {
        pipes.insert(name, mk(name, &tree_pipeline_layout));
    }

    // --- Buffers ---
    let buffer = |label: &str, size: u64| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: PARAM_SLOTS * PARAM_STRIDE,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let particles_buf = buffer("particles", plans.count as u64 * PARTICLE_BYTES);
    let moments_buf = buffer("moments", plans.cells as u64 * MOMENT_BYTES);
    let fields_buf = buffer("fields", plans.cells as u64 * FIELD_BYTES);

    for (l, p) in plans.level.iter().enumerate() {
        queue.write_buffer(&params_buf, l as u64 * PARAM_STRIDE, p);
    }
    for (l, p) in plans.upward.iter().enumerate() {
        queue.write_buffer(&params_buf, (8 + l as u64) * PARAM_STRIDE, p);
    }

    let params_binding = |offset: u64| {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &params_buf,
            offset,
            size: wgpu::BufferSize::new(PARAM_BYTES),
        })
    };
    let compute_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute"),
        layout: &compute_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_binding(0) },
            wgpu::BindGroupEntry { binding: 1, resource: particles_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: moments_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: fields_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: moments_buf.as_entire_binding() },
        ],
    });
    let tree_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tree"),
        layout: &tree_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_binding(0) },
            wgpu::BindGroupEntry { binding: 1, resource: moments_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: fields_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: particles_buf.as_entire_binding() },
        ],
    });
    let empty_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("empty"),
        layout: &empty_layout,
        entries: &[],
    });

    let level_cells = |l: u32| 1u32 << (3 * l);
    // Particle passes use a 2D grid so budgets past 16.7M stay under the
    // per-dimension workgroup limit.
    let particle_grid = |n: u32| {
        (n.min(PARTICLE_ROW).div_ceil(256), n.div_ceil(PARTICLE_ROW))
    };
    let (depth, cells, count) = (plans.depth, plans.cells, plans.count);
    let base = depth * PARAM_STRIDE as u32;

    // --- Pass encoders ---
    let encode_clear = |cp: &mut wgpu::ComputePass| {
        cp.set_pipeline(&pipes["clear_cells"]);
        cp.set_bind_group(0, &compute_bind, &[base]);
        cp.dispatch_workgroups(cells.div_ceil(256), 1, 1);
    };
    let encode_deposit = |cp: &mut wgpu::ComputePass| {
        cp.set_pipeline(&pipes["deposit"]);
        cp.set_bind_group(0, &compute_bind, &[base]);
        let (gx, gy) = particle_grid(count);
        cp.dispatch_workgroups(gx, gy, 1);
    };
    let encode_upward = |cp: &mut wgpu::ComputePass| {
        cp.set_pipeline(&pipes["upward"]);
        for l in (0..depth).rev() {
            cp.set_bind_group(0, &compute_bind, &[(8 + l) * PARAM_STRIDE as u32]);
            cp.dispatch_workgroups(level_cells(l).div_ceil(64), 1, 1);
        }
    };
    let encode_m2l = |cp: &mut wgpu::ComputePass| {
        cp.set_pipeline(&pipes["m2l"]);
        for l in 1..=depth {
            cp.set_bind_group(3, &tree_bind, &[l * PARAM_STRIDE as u32]);
            cp.set_bind_group(1, &empty_bind, &[]);
            cp.set_bind_group(2, &empty_bind, &[]);
            cp.dispatch_workgroups(level_cells(l).div_ceil(64), 1, 1);
        }
    };
    let encode_downward = |cp: &mut wgpu::ComputePass| {
        cp.set_pipeline(&pipes["downward"]);
        for l in 1..=depth {
            cp.set_bind_group(3, &tree_bind, &[l * PARAM_STRIDE as u32]);
            cp.set_bind_group(1, &empty_bind, &[]);
            cp.set_bind_group(2, &empty_bind, &[]);
            cp.dispatch_workgroups(level_cells(l).div_ceil(64), 1, 1);
        }
    };
    let encode_near = |cp: &mut wgpu::ComputePass| {
        cp.set_pipeline(&pipes["near_field"]);
        for l in 1..=depth {
            cp.set_bind_group(3, &tree_bind, &[l * PARAM_STRIDE as u32]);
            cp.set_bind_group(1, &empty_bind, &[]);
            cp.set_bind_group(2, &empty_bind, &[]);
            cp.dispatch_workgroups(level_cells(l).div_ceil(64), 1, 1);
        }
    };
    let encode_integrate = |cp: &mut wgpu::ComputePass| {
        cp.set_pipeline(&pipes["integrate"]);
        cp.set_bind_group(1, &empty_bind, &[]);
        cp.set_bind_group(2, &empty_bind, &[]);
        cp.set_bind_group(3, &tree_bind, &[base]);
        let (gx, gy) = particle_grid(count);
        cp.dispatch_workgroups(gx, gy, 1);
    };
    // One solver frame: clear, deposit, upward, M2L, downward, near, integrate.
    let encode_frame = |enc: &mut wgpu::CommandEncoder| {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        encode_clear(&mut cp);
        encode_deposit(&mut cp);
        encode_upward(&mut cp);
        encode_m2l(&mut cp);
        encode_downward(&mut cp);
        encode_near(&mut cp);
        encode_integrate(&mut cp);
    };

    // --- Seed the particles once ---
    {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cp.set_pipeline(&pipes["initialize"]);
            cp.set_bind_group(0, &compute_bind, &[base]);
            let (gx, gy) = particle_grid(count);
            cp.dispatch_workgroups(gx, gy, 1);
        }
        queue.submit([enc.finish()]);
    }

    // --- Content digest of one field solve, for cross-checking with the browser ---
    {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            encode_clear(&mut cp);
            encode_deposit(&mut cp);
            encode_upward(&mut cp);
            encode_m2l(&mut cp);
            encode_downward(&mut cp);
            encode_near(&mut cp);
        }
        queue.submit([enc.finish()]);
    }
    println!(
        "fieldsHash={:08x} momentsHash={:08x}",
        digest(&device, &queue, &fields_buf, cells as u64 * FIELD_BYTES),
        digest(&device, &queue, &moments_buf, cells as u64 * MOMENT_BYTES),
    );

    // --- Warm up to reach boost clocks ---
    for _ in 0..30 {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for _ in 0..10 {
            encode_frame(&mut enc);
        }
        queue.submit([enc.finish()]);
    }
    device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();

    // --- True per-phase GPU time via timestamp queries, for one frame ---
    if adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
        let names = ["clear", "deposit", "upward", "m2l", "downward", "near", "integrate"];
        let n_ts = (names.len() * 2) as u32;
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: n_ts,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamps"),
            size: n_ts as u64 * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let phases: [&dyn Fn(&mut wgpu::ComputePass); 7] = [
            &encode_clear,
            &encode_deposit,
            &encode_upward,
            &encode_m2l,
            &encode_downward,
            &encode_near,
            &encode_integrate,
        ];
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for (i, phase) in phases.iter().enumerate() {
            let writes = wgpu::ComputePassTimestampWrites {
                query_set: &query_set,
                beginning_of_pass_write_index: Some((i * 2) as u32),
                end_of_pass_write_index: Some((i * 2 + 1) as u32),
            };
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(names[i]),
                timestamp_writes: Some(writes),
            });
            phase(&mut cp);
        }
        enc.resolve_query_set(&query_set, 0..n_ts, &resolve, 0);
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();

        let ticks = read_u32(&device, &queue, &resolve, n_ts as u64 * 8);
        // Timestamps are u64; read_u32 split each into two little-endian words.
        let period = queue.get_timestamp_period() as f64;
        let mut total = 0f64;
        for (i, name) in names.iter().enumerate() {
            let lo = ticks[i * 4] as u64 | (ticks[i * 4 + 1] as u64) << 32;
            let hi = ticks[i * 4 + 2] as u64 | (ticks[i * 4 + 3] as u64) << 32;
            let dt = hi.wrapping_sub(lo) as f64 * period / 1e6;
            total += dt;
            println!("  ts {:<10} {:.4} ms", name, dt);
        }
        println!("  ts TOTAL     {:.4} ms", total);
    }

    // --- Amortized wall-clock time of a full solve ---
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    for _ in 0..repeat {
        encode_frame(&mut enc);
    }
    let t0 = std::time::Instant::now();
    queue.submit([enc.finish()]);
    device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    println!(
        "full solve: {:.4} ms/step  ({} steps)",
        t0.elapsed().as_secs_f64() * 1e3 / repeat as f64,
        repeat
    );

    // --- Finiteness check over a particle slice ---
    let probe = count.min(8192) as u64;
    let words = read_u32(&device, &queue, &particles_buf, probe * PARTICLE_BYTES);
    let mut non_finite = 0usize;
    let mut max_abs = 0f32;
    for w in words {
        let f = f32::from_bits(w);
        if f.is_finite() {
            max_abs = max_abs.max(f.abs());
        } else {
            non_finite += 1;
        }
    }
    println!("non-finite={} maxAbs={:.6}", non_finite, max_abs);
}
