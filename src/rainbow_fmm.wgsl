// Rainbow FMM
// Sparse p=1 Cartesian fast multipole solver for WebGPU.
//
// The hierarchy is a bounded linear octree. Empty cells cost one count test.
// P2M and M2M create monopoles, M2L evaluates the classical 189-cell
// interaction stencil, and L2L translates acceleration + its symmetric
// Jacobian. The finest 27-cell neighborhood is evaluated separately.

const FIXED_POINT: f32 = 1024.0;
const TAU: f32 = 6.28318530718;
const ADAPTIVE_LEAF_CAPACITY: u32 = 16u;

struct Params {
    particle_count: u32,
    depth: u32,
    level: u32,
    level_offset: u32,

    level_dim: u32,
    parent_offset: u32,
    parent_dim: u32,
    total_cells: u32,

    dt: f32,
    gravity: f32,
    softening: f32,
    point_size: f32,

    time: f32,
    aspect: f32,
    camera_yaw: f32,
    camera_pitch: f32,

    viewport_width: f32,
    viewport_height: f32,
    exposure: f32,
    speed: f32,
}

struct Particle {
    pos_mass: vec4<f32>,
    vel_hue: vec4<f32>,
}

struct Moment {
    count: atomic<u32>,
    sum_x: atomic<i32>,
    sum_y: atomic<i32>,
    sum_z: atomic<i32>,
}

// a.xyz = acceleration, a.w = potential
// j0 = Jxx, Jxy, Jxz, Jyy
// j1 = Jyz, Jzz, reserved, reserved
struct Field {
    a: vec4<f32>,
    j0: vec4<f32>,
    j1: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> moments: array<Moment>;
@group(0) @binding(3) var<storage, read_write> fields: array<Field>;

@group(1) @binding(0) var<uniform> render_params: Params;
@group(1) @binding(1) var<storage, read> render_particles: array<Particle>;

fn zero_field() -> Field {
    return Field(
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
    );
}

fn add_field(a: Field, b: Field) -> Field {
    return Field(a.a + b.a, a.j0 + b.j0, a.j1 + b.j1);
}

fn field_j_mul(f: Field, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        f.j0.x * v.x + f.j0.y * v.y + f.j0.z * v.z,
        f.j0.y * v.x + f.j0.w * v.y + f.j1.x * v.z,
        f.j0.z * v.x + f.j1.x * v.y + f.j1.y * v.z,
    );
}

fn linear_index(c: vec3<u32>, dim: u32) -> u32 {
    return c.x + dim * (c.y + dim * c.z);
}

fn octree_offset(level: u32) -> u32 {
    return ((1u << (3u * level)) - 1u) / 7u;
}

fn cell_coord(index: u32, dim: u32) -> vec3<u32> {
    let plane = dim * dim;
    let z = index / plane;
    let rest = index - z * plane;
    let y = rest / dim;
    return vec3<u32>(rest - y * dim, y, z);
}

fn cell_center(c: vec3<u32>, dim: u32) -> vec3<f32> {
    return (vec3<f32>(c) + vec3<f32>(0.5)) * (2.0 / f32(dim)) - vec3<f32>(1.0);
}

fn position_cell(pos: vec3<f32>, dim: u32) -> vec3<u32> {
    let scaled = floor((clamp(pos, vec3<f32>(-0.99999), vec3<f32>(0.99999)) * 0.5 + 0.5) * f32(dim));
    return vec3<u32>(scaled);
}

fn moment_com(index: u32, count: u32) -> vec3<f32> {
    let inv = 1.0 / (FIXED_POINT * f32(count));
    return vec3<f32>(
        f32(atomicLoad(&moments[index].sum_x)),
        f32(atomicLoad(&moments[index].sum_y)),
        f32(atomicLoad(&moments[index].sum_z)),
    ) * inv;
}

fn target_is_on_adaptive_path(local_index: u32) -> bool {
    if (params.level == 0u) {
        return true;
    }
    let c = cell_coord(local_index, params.level_dim);
    let parent_cell = c / 2u;
    let parent_index = params.parent_offset +
        linear_index(parent_cell, params.parent_dim);
    return atomicLoad(&moments[parent_index].count) > ADAPTIVE_LEAF_CAPACITY;
}

fn target_is_adaptive_leaf(local_index: u32, count: u32) -> bool {
    return target_is_on_adaptive_path(local_index) &&
        (count <= ADAPTIVE_LEAF_CAPACITY || params.level == params.depth);
}

fn pair_field(source_index: u32, target_position: vec3<f32>, epsilon: f32) -> Field {
    let count = atomicLoad(&moments[source_index].count);
    if (count == 0u) {
        return zero_field();
    }

    let source = moment_com(source_index, count);
    let r = source - target_position;
    let r2 = dot(r, r) + epsilon * epsilon;
    let inv_r = inverseSqrt(r2);
    let inv_r3 = inv_r * inv_r * inv_r;
    let inv_r5 = inv_r3 * inv_r * inv_r;
    let mass = f32(count) / f32(params.particle_count);
    let gm = params.gravity * mass;
    let a = gm * r * inv_r3;

    let diagonal = -gm * inv_r3;
    let tidal = 3.0 * gm * inv_r5;
    return Field(
        vec4<f32>(a, -gm * inv_r),
        vec4<f32>(
            diagonal + tidal * r.x * r.x,
            tidal * r.x * r.y,
            tidal * r.x * r.z,
            diagonal + tidal * r.y * r.y,
        ),
        vec4<f32>(
            tidal * r.y * r.z,
            diagonal + tidal * r.z * r.z,
            0.0,
            0.0,
        ),
    );
}

fn hash32(value: u32) -> u32 {
    var x = value;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    return (x >> 16u) ^ x;
}

fn random01(seed: u32) -> f32 {
    return f32(hash32(seed)) * (1.0 / 4294967296.0);
}

@compute @workgroup_size(256)
fn initialize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) {
        return;
    }

    let u0 = random01(i * 11u + 1u);
    let u1 = random01(i * 11u + 2u);
    let u2 = random01(i * 11u + 3u);
    let u3 = random01(i * 11u + 4u);
    let u4 = random01(i * 11u + 5u);
    let u5 = random01(i * 11u + 6u);
    var pos: vec3<f32>;
    var vel: vec3<f32>;
    var hue: f32;

    if (u5 < 0.003) {
        // A faint spherical halo makes the 3D camera motion legible.
        let z = u0 * 2.0 - 1.0;
        let phi = TAU * u1;
        let radius = 0.78 * pow(u2, 0.3333333);
        let ring = sqrt(max(0.0, 1.0 - z * z));
        pos = radius * vec3<f32>(ring * cos(phi), z * 0.58, ring * sin(phi));
        vel = 0.035 * vec3<f32>(-pos.z, 0.0, pos.x);
        hue = fract(0.58 + 0.18 * z + u3 * 0.08);
    } else {
        // Four narrow logarithmic ribbons: dense enough to glow at a million
        // bodies, but separated enough to read as structure rather than fog.
        let radius = 0.055 + 0.61 * pow(u0, 0.72);
        let arm = f32(i & 3u) * (TAU / 4.0);
        let angle = arm + radius * 8.2 + (u1 - 0.5) * 0.055;
        let thickness = (u2 + u3 + u4 - 1.5) * (0.004 + radius * 0.012);
        let flat = vec3<f32>(
            radius * cos(angle),
            radius * sin(angle) * 0.72,
            thickness,
        );
        let tilt = 0.24;
        pos = vec3<f32>(
            flat.x,
            flat.y * cos(tilt) - flat.z * sin(tilt),
            flat.y * sin(tilt) + flat.z * cos(tilt),
        );

        let orbital = 0.105 / sqrt(radius + 0.055);
        vel = orbital * vec3<f32>(
            -sin(angle),
            cos(angle) * 0.72 * cos(tilt),
            cos(angle) * 0.72 * sin(tilt),
        );
        hue = fract(0.03 + radius * 1.45);
    }

    particles[i].pos_mass = vec4<f32>(pos, 1.0 / f32(params.particle_count));
    particles[i].vel_hue = vec4<f32>(vel, hue);
}

@compute @workgroup_size(256)
fn clear_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.total_cells) {
        return;
    }
    atomicStore(&moments[i].count, 0u);
    atomicStore(&moments[i].sum_x, 0);
    atomicStore(&moments[i].sum_y, 0);
    atomicStore(&moments[i].sum_z, 0);
    fields[i] = zero_field();
}

@compute @workgroup_size(256)
fn deposit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) {
        return;
    }
    let pos = particles[i].pos_mass.xyz;
    let c = position_cell(pos, params.level_dim);
    let cell = params.level_offset + linear_index(c, params.level_dim);
    atomicAdd(&moments[cell].count, 1u);
    atomicAdd(&moments[cell].sum_x, i32(round(pos.x * FIXED_POINT)));
    atomicAdd(&moments[cell].sum_y, i32(round(pos.y * FIXED_POINT)));
    atomicAdd(&moments[cell].sum_z, i32(round(pos.z * FIXED_POINT)));
}

@compute @workgroup_size(64)
fn upward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_index = gid.x;
    let cell_count = params.level_dim * params.level_dim * params.level_dim;
    if (local_index >= cell_count) {
        return;
    }
    let c = cell_coord(local_index, params.level_dim);
    var count = 0u;
    var sx = 0;
    var sy = 0;
    var sz = 0;

    for (var dz = 0u; dz < 2u; dz++) {
        for (var dy = 0u; dy < 2u; dy++) {
            for (var dx = 0u; dx < 2u; dx++) {
                let child_c = c * 2u + vec3<u32>(dx, dy, dz);
                let child = params.parent_offset + linear_index(child_c, params.parent_dim);
                count += atomicLoad(&moments[child].count);
                sx += atomicLoad(&moments[child].sum_x);
                sy += atomicLoad(&moments[child].sum_y);
                sz += atomicLoad(&moments[child].sum_z);
            }
        }
    }

    let target_index = params.level_offset + local_index;
    atomicStore(&moments[target_index].count, count);
    atomicStore(&moments[target_index].sum_x, sx);
    atomicStore(&moments[target_index].sum_y, sy);
    atomicStore(&moments[target_index].sum_z, sz);
}

@compute @workgroup_size(64)
fn m2l(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_index = gid.x;
    let cell_count = params.level_dim * params.level_dim * params.level_dim;
    if (local_index >= cell_count) {
        return;
    }

    let target_index = params.level_offset + local_index;
    if (atomicLoad(&moments[target_index].count) == 0u ||
        !target_is_on_adaptive_path(local_index)) {
        fields[target_index] = zero_field();
        return;
    }

    let tc_u = cell_coord(local_index, params.level_dim);
    let tc = vec3<i32>(tc_u);
    let parent = tc / 2;
    let target_center = cell_center(tc_u, params.level_dim);
    let dim_i = i32(params.level_dim);
    var result = zero_field();
    // The children of the 3x3x3 parent neighborhood, excluding the target's
    // immediate 3x3x3 neighborhood: at most 216 - 27 = 189 interactions.
    for (var pz = -1; pz <= 1; pz++) {
        for (var py = -1; py <= 1; py++) {
            for (var px = -1; px <= 1; px++) {
                let source_parent = parent + vec3<i32>(px, py, pz);
                if (any(source_parent < vec3<i32>(0)) ||
                    any(source_parent >= vec3<i32>(i32(params.parent_dim)))) {
                    continue;
                }
                for (var cz = 0; cz < 2; cz++) {
                    for (var cy = 0; cy < 2; cy++) {
                        for (var cx = 0; cx < 2; cx++) {
                            let source_c = source_parent * 2 + vec3<i32>(cx, cy, cz);
                            if (any(source_c < vec3<i32>(0)) ||
                                any(source_c >= vec3<i32>(dim_i))) {
                                continue;
                            }
                            let separation = abs(source_c - tc);
                            if (all(separation <= vec3<i32>(1))) {
                                continue;
                            }
                            let source_u = vec3<u32>(source_c);
                            let source_index = params.level_offset +
                                linear_index(source_u, params.level_dim);
                            let cell_epsilon = 0.06 / f32(params.level_dim);
                            result = add_field(
                                result,
                                pair_field(source_index, target_center, cell_epsilon),
                            );
                        }
                    }
                }
            }
        }
    }
    fields[target_index] = result;
}

@compute @workgroup_size(64)
fn downward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_index = gid.x;
    let cell_count = params.level_dim * params.level_dim * params.level_dim;
    if (local_index >= cell_count) {
        return;
    }
    let target_index = params.level_offset + local_index;
    if (atomicLoad(&moments[target_index].count) == 0u ||
        !target_is_on_adaptive_path(local_index)) {
        return;
    }

    let c = cell_coord(local_index, params.level_dim);
    let pc = c / 2u;
    let parent_index = params.parent_offset + linear_index(pc, params.parent_dim);
    let parent_field = fields[parent_index];
    let delta = cell_center(c, params.level_dim) - cell_center(pc, params.parent_dim);
    var inherited = parent_field;
    inherited.a = vec4<f32>(
        parent_field.a.xyz + field_j_mul(parent_field, delta),
        parent_field.a.w - dot(parent_field.a.xyz, delta),
    );
    fields[target_index] = add_field(fields[target_index], inherited);
}

@compute @workgroup_size(64)
fn near_field(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_index = gid.x;
    let cell_count = params.level_dim * params.level_dim * params.level_dim;
    if (local_index >= cell_count) {
        return;
    }
    let target_index = params.level_offset + local_index;
    let target_count = atomicLoad(&moments[target_index].count);
    if (target_count == 0u ||
        !target_is_adaptive_leaf(local_index, target_count)) {
        return;
    }

    let tc_u = cell_coord(local_index, params.level_dim);
    let tc = vec3<i32>(tc_u);
    let target_center = cell_center(tc_u, params.level_dim);
    let dim_i = i32(params.level_dim);
    var result = fields[target_index];
    let near_epsilon = max(params.softening, 0.72 / f32(params.level_dim));
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) {
                    continue;
                }
                let source_c = tc + vec3<i32>(dx, dy, dz);
                if (any(source_c < vec3<i32>(0)) ||
                    any(source_c >= vec3<i32>(dim_i))) {
                    continue;
                }
                let source_index = params.level_offset +
                    linear_index(vec3<u32>(source_c), params.level_dim);
                result = add_field(
                    result,
                    pair_field(source_index, target_center, near_epsilon),
                );
            }
        }
    }
    fields[target_index] = result;
}

@compute @workgroup_size(256)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) {
        return;
    }
    var particle = particles[i];
    var pos = particle.pos_mass.xyz;
    var velocity = particle.vel_hue.xyz;
    // Descend only while the local density requires it. This selects a true
    // non-uniform leaf per particle without a serial tree walk or dynamic
    // allocation: the maximum six checks are a bounded constant.
    var chosen_dim = params.level_dim;
    var chosen_cell = position_cell(pos, chosen_dim);
    var cell_index = params.level_offset + linear_index(chosen_cell, chosen_dim);
    for (var level = 1u; level <= params.depth; level++) {
        let dim = 1u << level;
        let c = position_cell(pos, dim);
        let index = octree_offset(level) + linear_index(c, dim);
        let count_at_level = atomicLoad(&moments[index].count);
        if (count_at_level <= ADAPTIVE_LEAF_CAPACITY || level == params.depth) {
            chosen_dim = dim;
            chosen_cell = c;
            cell_index = index;
            break;
        }
    }
    let center = cell_center(chosen_cell, chosen_dim);
    let field = fields[cell_index];
    var acceleration = field.a.xyz + field_j_mul(field, pos - center);

    // Approximate the unresolved content of the particle's own leaf with a
    // softened monopole about its center of mass. This is bounded and O(N).
    let count = atomicLoad(&moments[cell_index].count);
    if (count > 1u) {
        let com = moment_com(cell_index, count);
        let delta = com - pos;
        let epsilon = max(params.softening, 0.62 / f32(chosen_dim));
        let inv_r = inverseSqrt(dot(delta, delta) + epsilon * epsilon);
        let self_mass = f32(count - 1u) / f32(params.particle_count);
        acceleration += params.gravity * self_mass * delta * inv_r * inv_r * inv_r;
    }

    // One-force symplectic Euler: unlike explicit Euler, energy error remains
    // bounded for the smooth Hamiltonian used by the visual simulation.
    let dt = params.dt * params.speed;
    velocity += acceleration * dt;
    pos += velocity * dt;

    // A safety boundary only affects escaped particles. Normal presets stay
    // well inside it; it prevents fixed-point overflow after extreme tweaking.
    for (var axis = 0u; axis < 3u; axis++) {
        if (abs(pos[axis]) > 0.995) {
            pos[axis] = clamp(pos[axis], -0.995, 0.995);
            velocity[axis] *= -0.82;
        }
    }

    particle.pos_mass = vec4<f32>(pos, particle.pos_mass.w);
    particle.vel_hue = vec4<f32>(velocity, particle.vel_hue.w);
    particles[i] = particle;
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) depth_fade: f32,
}

fn rainbow(hue: f32) -> vec3<f32> {
    let phase = vec3<f32>(0.00, 0.67, 0.33);
    return 0.5 + 0.5 * cos(TAU * (hue + phase));
}

fn rotate_camera(p: vec3<f32>, yaw: f32, pitch: f32) -> vec3<f32> {
    let cy = cos(yaw);
    let sy = sin(yaw);
    let cp = cos(pitch);
    let sp = sin(pitch);
    let y_rotated = vec3<f32>(cy * p.x + sy * p.z, p.y, -sy * p.x + cy * p.z);
    return vec3<f32>(
        y_rotated.x,
        cp * y_rotated.y - sp * y_rotated.z,
        sp * y_rotated.y + cp * y_rotated.z,
    );
}

@vertex
fn render_vertex(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );
    let particle_index = (
        instance_index * 747796405u + 2891336453u
    ) % render_params.particle_count;
    let particle = render_particles[particle_index];
    let rotated = rotate_camera(
        particle.pos_mass.xyz,
        render_params.camera_yaw,
        render_params.camera_pitch,
    );
    let view_z = render_params.speed - rotated.z;
    let perspective = 1.92 / view_z;
    let center = vec2<f32>(
        rotated.x * perspective / render_params.aspect,
        rotated.y * perspective,
    );
    let velocity = length(particle.vel_hue.xyz);
    let depth_size = clamp(perspective * 0.72, 0.48, 1.35);
    let size = render_params.point_size * depth_size * (0.9 + min(velocity, 0.5) * 0.7);
    let corner = corners[vertex_index];
    let pixel = vec2<f32>(
        2.0 * size / render_params.viewport_width,
        2.0 * size / render_params.viewport_height,
    );
    let world = particle.pos_mass.xyz;
    let spatial_phase = atan2(world.y, world.x) / TAU + length(world) * 0.31;
    let hue = fract(0.54 + spatial_phase + velocity * 0.035 + render_params.time * 0.0015);

    var out: VertexOutput;
    out.position = vec4<f32>(center + corner * pixel, 0.46 + rotated.z * 0.08, 1.0);
    out.uv = corner;
    out.color = rainbow(hue);
    out.depth_fade = clamp(0.62 + rotated.z * 0.25, 0.28, 1.0);
    return out;
}

@fragment
fn render_fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let radius2 = dot(in.uv, in.uv);
    if (radius2 > 1.0) {
        discard;
    }
    let halo = exp(-radius2 * 3.0) * 0.36;
    let core = exp(-radius2 * 18.0) * 1.25;
    // Normalize against the visible sample, not the million simulated bodies.
    // The sampled pass stays bright while preserving a bounded fragment cost.
    let population_scale = 0.65 * pow(
        16384.0 / max(f32(render_params.level), 1.0),
        0.38,
    );
    let intensity = (halo + core) * render_params.exposure * in.depth_fade * population_scale;
    return vec4<f32>(in.color * intensity, clamp(intensity, 0.0, 1.0));
}
