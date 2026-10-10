//! Rust-side planner and shader payload for the Rainbow FMM WebGPU demo.
//!
//! The browser shell deliberately stays small: this module owns the octree
//! layout, capacity policy, complexity estimates, and the WGSL implementation.
//! It is compiled as a dependency-free WebAssembly module.

const SHADER: &str = include_str!("rainbow_fmm.wgsl");
const ALGORITHM: &str = "Sparse p=1 Cartesian FMM / BLDTT-inspired adaptive hierarchy";

pub const MIN_DEPTH: u32 = 4;
pub const MAX_DEPTH: u32 = 6;
// Small plans are accepted for software-adapter diagnostics; the public UI
// starts at 2^17 and the tuned target remains 2^20.
pub const MIN_PARTICLES: u32 = 1 << 14;
pub const MAX_PARTICLES: u32 = 1 << 23;
// Keep about eight particles per finest cell. Refining past this only grows
// the (dense) octree footprint and hurts cache residency without improving the
// p=1 approximation.
pub const TARGET_PARTICLES_PER_LEAF: u32 = 8;
// The fixed-point moment sums are i32 atomics (WebGPU has no 64-bit atomics),
// so the deposit scale must shrink as the particle budget grows. Worst case is
// every body at the domain edge: sum = N * WORLD_HALF_EXTENT * FIXED_POINT.
pub const WORLD_HALF_EXTENT: f32 = 1.60;

/// Number of cells in one level of an octree.
pub const fn level_cells(level: u32) -> u32 {
    1u32 << (3 * level)
}

/// First linear cell index for a level.
pub const fn level_offset(level: u32) -> u32 {
    (level_cells(level) - 1) / 7
}

/// Total cells in levels `0..=depth`.
pub const fn total_cells(depth: u32) -> u32 {
    level_offset(depth + 1)
}

/// Choose a bounded leaf depth with roughly eight particles per finest cell.
///
/// The depth cap is important for WebGPU: it turns every per-level constant
/// into a true constant for the advertised particle range, and keeps the whole
/// octree (moments plus fields) small enough to stay cache-resident.
pub const fn depth_for_particles(particles: u32) -> u32 {
    let target_cells = particles.div_ceil(TARGET_PARTICLES_PER_LEAF);
    let mut depth = MIN_DEPTH;
    while depth < MAX_DEPTH && level_cells(depth) < target_cells {
        depth += 1;
    }
    depth
}

/// Fixed-point scale for the deposit pass.
///
/// Moment sums accumulate `round(pos * scale)` over up to `particles` bodies
/// inside one i32, so the scale is capped by the worst case of every body
/// sitting at the domain edge. A power of two keeps the division exact.
pub fn fixed_point_scale(particles: u32) -> f32 {
    // Positions are clamped below WORLD_HALF_EXTENT, so the worst-case sum is
    // particles * WORLD_HALF_EXTENT * scale. A headroom of 0.82 puts one
    // million bodies at their original scale of 1024 and every larger budget at
    // a safe power of two below it.
    const HEADROOM: f64 = 0.82;
    let worst = particles as f64 * WORLD_HALF_EXTENT as f64;
    let max_scale = HEADROOM * (i32::MAX as f64) / worst.max(1.0);
    let exponent = max_scale.log2().floor();
    (2.0f64).powf(exponent) as f32
}

pub const fn clamp_particle_count(requested: u32) -> u32 {
    if requested < MIN_PARTICLES {
        MIN_PARTICLES
    } else if requested > MAX_PARTICLES {
        MAX_PARTICLES
    } else {
        requested
    }
}

/// Work estimate for the complete solver. The M2L stencil has at most 189
/// source cells per occupied target cell; all other terms are linear scans.
pub const fn linear_work_bound(particles: u32, depth: u32) -> u64 {
    let cells = total_cells(depth) as u64;
    particles as u64 * 3 + cells * (189 + 8 + 2)
}

#[no_mangle]
pub extern "C" fn rainbow_api_version() -> u32 {
    1
}

#[no_mangle]
pub extern "C" fn rainbow_shader_ptr() -> *const u8 {
    SHADER.as_ptr()
}

#[no_mangle]
pub extern "C" fn rainbow_shader_len() -> usize {
    SHADER.len()
}

#[no_mangle]
pub extern "C" fn rainbow_algorithm_ptr() -> *const u8 {
    ALGORITHM.as_ptr()
}

#[no_mangle]
pub extern "C" fn rainbow_algorithm_len() -> usize {
    ALGORITHM.len()
}

#[no_mangle]
pub extern "C" fn rainbow_depth_for_particles(particles: u32) -> u32 {
    depth_for_particles(clamp_particle_count(particles))
}

#[no_mangle]
pub extern "C" fn rainbow_level_cells(level: u32) -> u32 {
    if level <= MAX_DEPTH {
        level_cells(level)
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn rainbow_level_offset(level: u32) -> u32 {
    if level <= MAX_DEPTH + 1 {
        level_offset(level)
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn rainbow_total_cells(depth: u32) -> u32 {
    total_cells(depth.clamp(MIN_DEPTH, MAX_DEPTH))
}

#[no_mangle]
pub extern "C" fn rainbow_clamp_particle_count(requested: u32) -> u32 {
    clamp_particle_count(requested)
}

#[no_mangle]
pub extern "C" fn rainbow_fixed_point_scale(particles: u32) -> f32 {
    fixed_point_scale(clamp_particle_count(particles))
}

#[no_mangle]
pub extern "C" fn rainbow_linear_work_lo(particles: u32, depth: u32) -> u32 {
    linear_work_bound(
        clamp_particle_count(particles),
        depth.clamp(MIN_DEPTH, MAX_DEPTH),
    ) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn octree_layout_is_contiguous() {
        for level in 0..=MAX_DEPTH {
            assert_eq!(
                level_offset(level) + level_cells(level),
                level_offset(level + 1)
            );
        }
        assert_eq!(total_cells(6), 299_593);
    }

    #[test]
    fn million_particle_plan_uses_depth_six() {
        assert_eq!(depth_for_particles(1 << 20), 6);
        assert_eq!(level_cells(6), 262_144);
        assert_eq!(clamp_particle_count(u32::MAX), 1 << 23);
    }

    #[test]
    fn multi_million_plans_stay_cache_resident() {
        // Depth six is the residency ceiling: its 4.8 MiB tree (moments plus
        // fields) fits on-die, while depth seven's 153 MiB does not.
        for particles in [1 << 21, 1 << 22, 1 << 23] {
            assert_eq!(depth_for_particles(particles), 6);
        }
        assert_eq!(total_cells(6), 299_593);
    }

    #[test]
    fn fixed_point_scale_stays_inside_i32() {
        for particles in [1 << 14, 1 << 20, 1 << 21, 1 << 22, 1 << 23] {
            let scale = fixed_point_scale(particles) as f64;
            let worst = particles as f64 * WORLD_HALF_EXTENT as f64 * scale;
            assert!(
                worst < i32::MAX as f64,
                "{particles} bodies overflow i32: {worst}"
            );
            // A power of two keeps the moment divide exact.
            assert_eq!(scale.log2().fract(), 0.0);
        }
    }

    #[test]
    fn work_bound_is_linear_for_a_fixed_depth() {
        let a = linear_work_bound(1 << 18, 6);
        let b = linear_work_bound(1 << 19, 6);
        assert_eq!(b - a, 3 * (1 << 18));
    }

    #[test]
    fn wasm_payload_is_present() {
        assert!(SHADER.contains("fn m2l"));
        assert!(SHADER.contains("fn integrate"));
        assert!(SHADER.len() > 10_000);
    }
}
