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
// A 60% wider half-extent gives escaped structures 4.1x the old volume while
// keeping worst-case fixed-point moment sums safely inside signed i32.
const WORLD_HALF_EXTENT: f32 = 1.60;
const WORLD_EDGE: f32 = WORLD_HALF_EXTENT * 0.985;

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

    // xyz = world-space brush center; w = active.
    control_point: vec4<f32>,
    // xyz = view-facing vortex axis; w = 1 pull, 2 push, 3 vortex.
    control_axis_mode: vec4<f32>,
    // x = acceleration, y = radius; zw reserved for diagnostics.
    control_settings: vec4<f32>,
}

struct Particle {
    // xyz = position; w = persistent dynamical hue.
    pos_light: vec4<f32>,
    // xyz = velocity; w = slowly decaying dynamical excitation.
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
@group(2) @binding(0) var accumulation_texture: texture_2d<f32>;
@group(2) @binding(1) var accumulation_sampler: sampler;

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
    return (
        (vec3<f32>(c) + vec3<f32>(0.5)) *
        (2.0 / f32(dim)) -
        vec3<f32>(1.0)
    ) * WORLD_HALF_EXTENT;
}

fn position_cell(pos: vec3<f32>, dim: u32) -> vec3<u32> {
    let normalized = clamp(
        pos / WORLD_HALF_EXTENT,
        vec3<f32>(-0.99999),
        vec3<f32>(0.99999),
    );
    let scaled = floor((normalized * 0.5 + 0.5) * f32(dim));
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

fn seeded_enclosed_mass(radius: f32) -> f32 {
    // The seeded population is 5% compact bulge, 70% uniform live halo,
    // 18% luminous disc, and 7% polar annulus.
    let bulge = 0.05 * pow(clamp(radius / 0.11, 0.0, 1.0), 3.0);
    let halo = 0.70 * pow(clamp(radius / 0.70, 0.0, 1.0), 3.0);
    let disc = 0.18 * pow(
        clamp((radius - 0.055) / 0.60, 0.0, 1.0),
        1.0 / 0.72,
    );
    let polar = 0.07 * pow(
        clamp((radius - 0.22) / 0.31, 0.0, 1.0),
        1.0 / 0.86,
    );
    return max(bulge + halo + disc + polar, 0.0001);
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
    var excitation: f32;

    if (u5 < 0.75) {
        // A live spherical mass component is the orbital backbone. Every halo
        // body still moves and contributes to FMM gravity, but is deliberately
        // faint so the stabilizer does not become a visual fog.
        let z = u0 * 2.0 - 1.0;
        let phi = TAU * u1;
        let outer_radius = select(0.70, 0.11, u5 < 0.05);
        let radius = outer_radius * pow(u2, 0.3333333);
        let ring = sqrt(max(0.0, 1.0 - z * z));
        let radial_direction = vec3<f32>(
            ring * cos(phi),
            z,
            ring * sin(phi),
        );
        pos = radius * radial_direction;

        let tangent_z = u3 * 2.0 - 1.0;
        let tangent_phi = TAU * u4;
        let tangent_ring = sqrt(max(0.0, 1.0 - tangent_z * tangent_z));
        let random_direction = vec3<f32>(
            tangent_ring * cos(tangent_phi),
            tangent_z,
            tangent_ring * sin(tangent_phi),
        );
        var tangent = cross(radial_direction, random_direction);
        if (length(tangent) < 0.001) {
            tangent = cross(radial_direction, vec3<f32>(0.0, 1.0, 0.0));
        }
        if (length(tangent) < 0.001) {
            tangent = cross(radial_direction, vec3<f32>(1.0, 0.0, 0.0));
        }
        let orbital = 0.985 * sqrt(
            params.gravity * seeded_enclosed_mass(radius) / (radius + 0.018),
        );
        vel = orbital * normalize(tangent);
        excitation = select(0.025 + u4 * 0.025, 0.09, u5 < 0.05);
    } else if (u5 < 0.93) {
        // A luminous four-arm tracer disc. The halo-dominated rotation curve
        // has much less differential shear, so its density waves survive.
        let radius = 0.055 + 0.60 * pow(u0, 0.72);
        let arm = f32(i & 3u) * (TAU / 4.0);
        let angle = arm + radius * 7.8 +
            (u1 - 0.5) * (0.045 + radius * 0.055);
        let thickness = (u2 + u3 + u4 - 1.5) *
            (0.0035 + radius * 0.010);
        let flat = vec3<f32>(
            radius * cos(angle),
            radius * sin(angle) * 0.72,
            thickness,
        );
        let tilt = 0.20;
        pos = vec3<f32>(
            flat.x,
            flat.y * cos(tilt) - flat.z * sin(tilt),
            flat.y * sin(tilt) + flat.z * cos(tilt),
        );

        let orbital = 0.99 * sqrt(
            params.gravity * seeded_enclosed_mass(radius) / (radius + 0.018),
        );
        let tangent = vec3<f32>(
            -sin(angle),
            cos(angle) * 0.72 * cos(tilt),
            cos(angle) * 0.72 * sin(tilt),
        );
        let radial = normalize(pos) * orbital * 0.035 *
            sin(angle * 3.0 - radius * 13.0);
        vel = orbital * normalize(tangent) + radial;
        excitation = 0.08 + u3 * 0.05;
    } else {
        // Four retrograde polar lanes repeatedly cross the main disc. Each
        // lineage owns a slightly different plane.
        let lineage = f32(i & 3u);
        let radius =
            0.22 + 0.31 * pow(u0, 0.86) + (lineage - 1.5) * 0.007;
        let angle = TAU * u1 + radius * 3.5;
        let thickness = (u2 + u3 + u4 - 1.5) *
            (0.004 + radius * 0.009);
        let flat = vec3<f32>(
            radius * cos(angle),
            thickness,
            radius * sin(angle),
        );
        let tilt = -0.54 + lineage * 0.36;
        pos = vec3<f32>(
            flat.x * cos(tilt) - flat.y * sin(tilt),
            flat.x * sin(tilt) + flat.y * cos(tilt),
            flat.z,
        );

        let tangent = vec3<f32>(
            -sin(angle) * cos(tilt),
            -sin(angle) * sin(tilt),
            cos(angle),
        );
        let orbital = 0.99 * sqrt(
            params.gravity * seeded_enclosed_mass(radius) / (radius + 0.018),
        );
        vel = -orbital * normalize(tangent);
        excitation = 0.20 + u4 * 0.08;
    }

    var dynamical_hue = 0.0;
    if (u5 >= 0.75) {
        let speed = max(length(vel), 0.0001);
        let radius = max(length(pos), 0.0001);
        let radial_motion = clamp(
            dot(pos, vel) / (radius * speed),
            -1.0,
            1.0,
        );
        let angular_momentum = cross(pos, vel);
        let spin_signature = angular_momentum.y /
            max(length(angular_momentum), 0.0001);
        dynamical_hue = fract(
            atan2(vel.y, vel.x) / TAU +
            (vel.z / speed) * 0.14 +
            spin_signature * 0.075 +
            radial_motion * 0.065 +
            excitation * 0.085 +
            f32(i & 3u) * 0.018
        );
    }
    particles[i].pos_light = vec4<f32>(pos, dynamical_hue);
    particles[i].vel_hue = vec4<f32>(vel, excitation);
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
    let pos = particles[i].pos_light.xyz;
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
                            let cell_epsilon =
                                0.06 * WORLD_HALF_EXTENT /
                                f32(params.level_dim);
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
    let near_epsilon = max(
        params.softening,
        0.72 * WORLD_HALF_EXTENT / f32(params.level_dim),
    );
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
    var pos = particle.pos_light.xyz;
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
        let epsilon = max(
            params.softening,
            0.62 * WORLD_HALF_EXTENT / f32(chosen_dim),
        );
        let inv_r = inverseSqrt(dot(delta, delta) + epsilon * epsilon);
        let self_mass = f32(count - 1u) / f32(params.particle_count);
        acceleration += params.gravity * self_mass * delta * inv_r * inv_r * inv_r;
    }

    // The interactive shepherd is a smooth, finite-radius external field.
    // It is one bounded calculation per body, so it does not change O(N).
    var control_influence = 0.0;
    if (params.control_point.w > 0.5) {
        let control_delta = params.control_point.xyz - pos;
        let distance2 = dot(control_delta, control_delta);
        let control_radius = max(params.control_settings.y, 0.02);
        control_influence = exp(
            -2.0 * distance2 / (control_radius * control_radius),
        );
        let control_strength = params.control_settings.x;
        let direction = control_delta * inverseSqrt(
            distance2 + control_radius * control_radius * 0.018,
        );
        let mode = params.control_axis_mode.w;
        if (mode < 1.5) {
            acceleration += direction * control_strength * control_influence;
        } else if (mode < 2.5) {
            acceleration -= direction * control_strength * control_influence;
        } else {
            let tangent = cross(
                normalize(params.control_axis_mode.xyz),
                control_delta,
            );
            let tangent_length = length(tangent);
            if (tangent_length > 0.00001) {
                acceleration += tangent / tangent_length *
                    control_strength * control_influence;
            }
        }
    }

    // One-force symplectic Euler: unlike explicit Euler, energy error remains
    // bounded for the smooth Hamiltonian used by the visual simulation.
    let dt = params.dt * params.speed;
    let radius = max(length(pos), 0.0001);
    let velocity_length = max(length(velocity), 0.0001);
    let signed_radial_motion = clamp(
        dot(pos, velocity) / (radius * velocity_length),
        -1.0,
        1.0,
    );
    let radial_fraction = abs(signed_radial_motion);
    let radial_shock = radial_fraction * radial_fraction * radial_fraction;
    let density_pulse = clamp(
        (f32(count) - f32(ADAPTIVE_LEAF_CAPACITY)) / 72.0,
        0.0,
        1.0,
    );
    let field_strength = clamp(
        length(acceleration) * (radius + 0.12) * 0.58,
        0.0,
        1.0,
    );
    let target_excitation = clamp(
        0.025 +
        radial_shock * 0.56 +
        density_pulse * 0.26 +
        field_strength * 0.13 +
        control_influence * 0.42,
        0.0,
        1.0,
    );
    var excitation = particle.vel_hue.w;
    // Compression and radial shocks light up quickly; the state cools slowly,
    // leaving a short physical memory instead of coloring absolute position.
    let response = select(1.05, 10.0, target_excitation > excitation);
    excitation = mix(
        excitation,
        target_excitation,
        clamp(dt * response, 0.0, 1.0),
    );

    velocity += acceleration * dt;
    pos += velocity * dt;

    // A safety boundary only affects particles that escape the expanded FMM
    // domain. It prevents fixed-point overflow after extreme tweaking.
    for (var axis = 0u; axis < 3u; axis++) {
        if (abs(pos[axis]) > WORLD_EDGE) {
            pos[axis] = clamp(pos[axis], -WORLD_EDGE, WORLD_EDGE);
            velocity[axis] *= -0.82;
        }
    }

    // Hue is a persistent material state seeded from orbital phase. Shocks and
    // external work advect it slowly, so color reacts without a render-time
    // coordinate lookup or expensive per-vertex inverse trigonometry.
    let dynamical_hue = fract(
        particle.pos_light.w +
        dt * (
            0.012 +
            signed_radial_motion * 0.030 +
            excitation * 0.036 +
            control_influence * 0.080
        )
    );
    particle.pos_light = vec4<f32>(pos, dynamical_hue);
    particle.vel_hue = vec4<f32>(velocity, excitation);
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

fn rotate_camera(p: vec3<f32>) -> vec3<f32> {
    // Render plans cache these four trigonometric values in compute-only
    // parameter slots, avoiding four transcendental evaluations per body.
    let cy = render_params.dt;
    let sy = render_params.gravity;
    let cp = render_params.softening;
    let sp = render_params.control_settings.z;
    let y_rotated = vec3<f32>(cy * p.x + sy * p.z, p.y, -sy * p.x + cy * p.z);
    return vec3<f32>(
        y_rotated.x,
        cp * y_rotated.y - sp * y_rotated.z,
        sp * y_rotated.y + cp * y_rotated.z,
    );
}

@vertex
fn render_vertex(
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let particle_index = (
        instance_index * 747796405u + 2891336453u
    ) % render_params.particle_count;
    let particle = render_particles[particle_index];
    let rotated = rotate_camera(particle.pos_light.xyz);
    // Move the camera back with the larger domain while retaining the original
    // on-screen scale for the unchanged central seed.
    let view_z =
        render_params.speed * WORLD_HALF_EXTENT -
        rotated.z;
    let perspective =
        1.92 * WORLD_HALF_EXTENT /
        view_z;
    let center = vec2<f32>(
        rotated.x * perspective / render_params.aspect,
        rotated.y * perspective,
    );
    let world = particle.pos_light.xyz;
    let world_velocity = particle.vel_hue.xyz;
    let velocity = length(world_velocity);
    let speed = max(velocity, 0.0001);
    let excitation = clamp(particle.vel_hue.w, 0.0, 1.0);
    let family = random01(particle_index * 11u + 6u);
    var luminosity = 1.0;
    var spectral = vec3<f32>(1.00, 0.66, 0.30);
    if (family < 0.05) {
        luminosity = 0.10;
        spectral = vec3<f32>(1.00, 0.66, 0.30);
    } else if (family < 0.75) {
        luminosity = 0.032;
        // Keep the numerous, faint mass halo quiet and cool. Its orbital-plane
        // signature changes value, not hue, so it cannot become confetti.
        spectral = vec3<f32>(0.12, 0.24, 0.38) * (
            0.72 + (world_velocity.y / speed * 0.5 + 0.5) * 0.34
        );
    } else {
        if (family >= 0.93) {
            luminosity = 1.22;
        }
        spectral = pow(
            rainbow(particle.pos_light.w),
            vec3<f32>(0.88),
        );
    }
    // Phase-space state controls both the continuous spectrum and luminance.
    // Absolute position does not participate.
    let state_gain =
        0.80 +
        min(velocity, 1.0) * 0.10 +
        excitation * 0.26;
    let heated = min(
        vec3<f32>(1.0),
        spectral * state_gain +
        vec3<f32>(excitation * excitation * 0.06),
    );
    var control_influence = 0.0;
    var control_color = vec3<f32>(0.82, 1.0, 0.28);
    if (render_params.control_point.w > 0.5) {
        let control_delta = render_params.control_point.xyz - world;
        let control_radius = max(render_params.control_settings.y, 0.02);
        control_influence = exp(
            -2.0 * dot(control_delta, control_delta) /
            (control_radius * control_radius),
        );
        let mode = render_params.control_axis_mode.w;
        if (mode > 1.5 && mode < 2.5) {
            control_color = vec3<f32>(1.0, 0.38, 0.18);
        } else if (mode >= 2.5) {
            control_color = vec3<f32>(0.82, 0.34, 1.0);
        }
    }
    let display_color = mix(
        heated,
        control_color,
        control_influence * 0.68,
    );

    var out: VertexOutput;
    out.position = vec4<f32>(
        center,
        0.46 + rotated.z * 0.08,
        1.0,
    );
    out.uv = vec2<f32>(0.0);
    out.color = display_color;
    out.depth_fade =
        clamp(0.62 + rotated.z * 0.25, 0.28, 1.0) *
        (0.76 + excitation * 0.48) *
        luminosity *
        (1.0 + control_influence * 0.72);
    return out;
}

@fragment
fn render_fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let population_scale = 0.62 * pow(
        1048576.0 / max(f32(render_params.level), 1.0),
        0.14,
    );
    let weight =
        render_params.exposure * in.depth_fade * population_scale;
    // RGB is premultiplied color energy; alpha is density. Both add without
    // clamping in rgba16float.
    return vec4<f32>(in.color * weight, weight);
}

@vertex
fn resolve_vertex(
    @builtin(vertex_index) vertex_index: u32,
) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(positions[vertex_index], 0.0, 1.0);
}

fn reconstruct_field(uv: vec2<f32>, axis: vec2<f32>) -> vec4<f32> {
    // point_size is the requested outer Gaussian radius in final-canvas
    // pixels. The furthest optimized tap lands exactly at that radius.
    let output_dimensions = vec2<f32>(
        render_params.viewport_width,
        render_params.viewport_height,
    );
    let step = axis *
        max(render_params.point_size, 1.0) /
        (output_dimensions * 3.230769);
    var field = textureSample(
        accumulation_texture, accumulation_sampler, uv,
    ) * 0.227027;
    field += textureSample(
        accumulation_texture,
        accumulation_sampler,
        uv + step * 1.384615,
    ) * 0.316216;
    field += textureSample(
        accumulation_texture,
        accumulation_sampler,
        uv - step * 1.384615,
    ) * 0.316216;
    field += textureSample(
        accumulation_texture,
        accumulation_sampler,
        uv + step * 3.230769,
    ) * 0.070270;
    field += textureSample(
        accumulation_texture,
        accumulation_sampler,
        uv - step * 3.230769,
    ) * 0.070270;
    return field;
}

@fragment
fn blur_horizontal_fragment(
    @builtin(position) position: vec4<f32>,
) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(accumulation_texture));
    let uv = position.xy / dimensions;
    return reconstruct_field(uv, vec2<f32>(1.0, 0.0));
}

@fragment
fn blur_vertical_fragment(
    @builtin(position) position: vec4<f32>,
) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(accumulation_texture));
    let uv = position.xy / dimensions;
    return reconstruct_field(uv, vec2<f32>(0.0, 1.0));
}

@fragment
fn resolve_fragment(
    @builtin(position) position: vec4<f32>,
) -> @location(0) vec4<f32> {
    let uv = position.xy / vec2<f32>(
        render_params.viewport_width,
        render_params.viewport_height,
    );
    let accumulated = textureSample(
        accumulation_texture,
        accumulation_sampler,
        uv,
    );
    let background = vec3<f32>(0.006, 0.008, 0.015);
    if (accumulated.a < 0.00001) {
        return vec4<f32>(background, 1.0);
    }

    // Density normalization makes a red cloud remain red whether one or one
    // hundred samples overlap. The density controls opacity separately.
    let average = accumulated.rgb / accumulated.a;
    let luma = dot(average, vec3<f32>(0.2126, 0.7152, 0.0722));
    let vivid = max(
        vec3<f32>(0.0),
        mix(vec3<f32>(luma), average, 1.16),
    );
    let density = sqrt(accumulated.a / (accumulated.a + 1.0));
    let mapped = vec3<f32>(1.0) - exp(-vivid * density * 1.58);
    let display = pow(clamp(mapped + background, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(0.82));
    return vec4<f32>(display, 1.0);
}
