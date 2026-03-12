// Copyright (c) 2026 Omair Kamil oakamil@gmail.com
// See LICENSE file in root directory for license terms.

use kiddo::{KdTree, SquaredEuclidean};
use nalgebra::{DMatrix, DVector, Matrix3, SVD};
use ndarray::{Array1, Array2};
use npyz::NpyFile;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use zip::ZipArchive;

const MAGIC_RAND: u64 = 2654435761;

// --- Data Structures & Options ---

#[derive(Debug, Clone, PartialEq, Default)]
pub enum SolveStatus {
    MatchFound,
    #[default]
    NoMatch,
    Timeout,
    Cancelled,
    TooFew,
}

#[derive(Debug, Clone)]
pub struct SolveOptions {
    pub fov_estimate: Option<f64>,
    pub fov_max_error: Option<f64>,
    pub match_radius: f64,
    pub match_threshold: f64,
    pub solve_timeout_ms: Option<f64>,
    pub distortion: Option<f64>,
    pub match_max_error: f64,
    pub return_matches: bool,
    pub return_catalog: bool,
    pub return_rotation_matrix: bool,
    pub target_pixel: Option<Array2<f64>>,     // N x 2 (y, x)
    pub target_sky_coord: Option<Array2<f64>>, // N x 2 (ra, dec)
}

impl Default for SolveOptions {
    fn default() -> Self {
        SolveOptions {
            fov_estimate: None,
            fov_max_error: None,
            match_radius: 0.01,
            match_threshold: 1e-5,
            solve_timeout_ms: Some(5000.0),
            distortion: None,
            match_max_error: 0.002,
            return_matches: false,
            return_catalog: false,
            return_rotation_matrix: false,
            target_pixel: None,
            target_sky_coord: None,
        }
    }
}

#[derive(Debug, Default)]
pub struct Solution {
    pub ra: Option<f64>,
    pub dec: Option<f64>,
    pub roll: Option<f64>,
    pub fov: Option<f64>,
    pub distortion: Option<f64>,
    pub rmse: Option<f64>,
    pub p90e: Option<f64>,
    pub maxe: Option<f64>,
    pub matches: Option<usize>,
    pub prob: Option<f64>,
    pub epoch_equinox: Option<f64>,
    pub epoch_proper_motion: Option<f64>,
    pub status: SolveStatus,
    pub t_solve_ms: f64,
    pub rotation_matrix: Option<Array2<f64>>,
    pub target_ra: Option<Vec<f64>>,
    pub target_dec: Option<Vec<f64>>,
    pub target_y: Option<Vec<Option<f64>>>,
    pub target_x: Option<Vec<Option<f64>>>,
    pub matched_centroids: Option<Vec<[f64; 2]>>,
    pub matched_stars: Option<Vec<[f64; 3]>>, // ra, dec, mag
    pub matched_cat_id: Option<Vec<u32>>,
    pub catalog_stars: Option<Vec<(f64, f64, f64, f64, f64)>>, // ra, dec, mag, y, x
}

#[derive(Clone, Copy)]
pub struct CatalogStar {
    pub ra: f64,
    pub dec: f64,
    pub vec: [f64; 3],
    pub mag: f64,
}

// --- High-Performance Native Math Helpers ---

#[inline(always)]
fn angle_from_distance(dist: f64) -> f64 {
    2.0 * (0.5 * dist).asin()
}

#[inline(always)]
fn distance_from_angle(angle: f64) -> f64 {
    2.0 * (angle / 2.0).sin()
}

// OPTIMIZATION: Replaces incredibly heavy `statrs::Binomial` cdf calculation inside hot loops.
// Nanosecond execution time for small N probabilities, exactly mirroring scipy.stats.binom.cdf.
fn fast_binomial_cdf(k: i64, n: u64, p: f64) -> f64 {
    if k < 0 {
        return 0.0;
    }
    if k >= n as i64 {
        return 1.0;
    }
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return 0.0;
    }

    let mut cdf = 0.0;
    let mut term = (1.0 - p).powi(n as i32);
    cdf += term;
    for i in 1..=(k as u64) {
        term = term * (n - i + 1) as f64 / i as f64 * p / (1.0 - p);
        cdf += term;
    }
    cdf
}

fn compute_vectors_flat(
    centroids: &[[f64; 2]],
    height: f64,
    width: f64,
    fov: f64,
) -> Vec<[f64; 3]> {
    let scale_factor = (fov / 2.0).tan() / width * 2.0;
    let img_center_y = height / 2.0;
    let img_center_x = width / 2.0;
    let mut out = Vec::with_capacity(centroids.len());

    for c in centroids {
        let v0 = 1.0;
        let v1 = (img_center_x - c[1]) * scale_factor;
        let v2 = (img_center_y - c[0]) * scale_factor;
        let norm = (v0 * v0 + v1 * v1 + v2 * v2).sqrt();
        out.push([v0 / norm, v1 / norm, v2 / norm]);
    }
    out
}

// OPTIMIZATION: Zero-allocation inner loop alternative for compute_vectors
fn compute_vectors_inplace(
    centroids: &[[f64; 2]],
    height: f64,
    width: f64,
    fov: f64,
    out: &mut [[f64; 3]],
    len: usize,
) {
    let scale_factor = (fov / 2.0).tan() / width * 2.0;
    let img_center_y = height / 2.0;
    let img_center_x = width / 2.0;

    for i in 0..len {
        let v0 = 1.0;
        let v1 = (img_center_x - centroids[i][1]) * scale_factor;
        let v2 = (img_center_y - centroids[i][0]) * scale_factor;
        let norm = (v0 * v0 + v1 * v1 + v2 * v2).sqrt();
        out[i] = [v0 / norm, v1 / norm, v2 / norm];
    }
}

// OPTIMIZATION: Zero-allocation inner loop alternative for compute_centroids
fn compute_centroids_inplace(
    vectors: &[[f64; 3]],
    height: f64,
    width: f64,
    fov: f64,
    out_centroids: &mut [[f64; 2]],
    out_kept: &mut Vec<usize>,
    len: usize,
) {
    out_kept.clear();
    let scale_factor = -width / 2.0 / (fov / 2.0).tan();
    let img_center_y = height / 2.0;
    let img_center_x = width / 2.0;

    for i in 0..len {
        let cy = scale_factor * (vectors[i][2] / vectors[i][0]) + img_center_y;
        let cx = scale_factor * (vectors[i][1] / vectors[i][0]) + img_center_x;
        out_centroids[i][0] = cy;
        out_centroids[i][1] = cx;

        if cy > 0.0 && cx > 0.0 && cy < height && cx < width {
            out_kept.push(i);
        }
    }
}

fn undistort_centroids(centroids: &[[f64; 2]], height: f64, width: f64, k: f64) -> Vec<[f64; 2]> {
    let kp = k * (2.0 / width).powi(2);
    let mut undistorted = centroids.to_vec();

    for row in undistorted.iter_mut() {
        let dy = row[0] - height / 2.0;
        let dx = row[1] - width / 2.0;
        let r_dist = (dy * dy + dx * dx).sqrt();
        let scale = (1.0 - kp * r_dist.powi(2)) / (1.0 - k);
        row[0] = (dy * scale) + height / 2.0;
        row[1] = (dx * scale) + width / 2.0;
    }
    undistorted
}

fn distort_centroids(
    centroids: &[[f64; 2]],
    height: f64,
    width: f64,
    k: f64,
    tol: f64,
    maxiter: usize,
) -> Vec<[f64; 2]> {
    let kp = k * (2.0 / width).powi(2);
    let mut distorted = centroids.to_vec();

    for row in distorted.iter_mut() {
        let dy = row[0] - height / 2.0;
        let dx = row[1] - width / 2.0;
        let r_undist = (dy * dy + dx * dx).sqrt();

        if r_undist < 1e-8 {
            continue;
        }

        let mut r_dist = r_undist;
        for _ in 0..maxiter {
            let r_undist_est = r_dist * (1.0 - kp * r_dist.powi(2)) / (1.0 - k);
            let dru_drd = (1.0 - 2.0 * kp * r_dist) / (1.0 - k);
            let error = r_undist - r_undist_est;
            r_dist += error / dru_drd;
            if error.abs() < tol {
                break;
            }
        }
        let scale = r_dist / r_undist;
        row[0] = (dy * scale) + height / 2.0;
        row[1] = (dx * scale) + width / 2.0;
    }
    distorted
}

// OPTIMIZATION: Zero-allocation inner loop alternative for sort_vectors_by_radius
fn sort_vectors_by_radius_inplace(
    vectors: &[[f64; 3]],
    sorted_out: &mut [[f64; 3]],
    radii_scratch: &mut Vec<(f64, usize)>,
    len: usize,
) {
    let mut centroid = [0.0, 0.0, 0.0];
    for i in 0..len {
        centroid[0] += vectors[i][0];
        centroid[1] += vectors[i][1];
        centroid[2] += vectors[i][2];
    }
    centroid[0] /= len as f64;
    centroid[1] /= len as f64;
    centroid[2] /= len as f64;

    radii_scratch.clear();
    for i in 0..len {
        let dx = vectors[i][0] - centroid[0];
        let dy = vectors[i][1] - centroid[1];
        let dz = vectors[i][2] - centroid[2];
        radii_scratch.push(((dx * dx + dy * dy + dz * dz).sqrt(), i));
    }

    radii_scratch.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    for (new_idx, &(_, old_idx)) in radii_scratch.iter().enumerate() {
        sorted_out[new_idx] = vectors[old_idx];
    }
}

// OPTIMIZATION: Fully unrolled, zero-allocation rotation helper.
fn rotate_vectors_inplace(
    rot: &Matrix3<f64>,
    vecs: &[[f64; 3]],
    transpose_rot: bool,
    out: &mut [[f64; 3]],
    len: usize,
) {
    let r = if transpose_rot { rot.transpose() } else { *rot };
    for i in 0..len {
        out[i][0] = r[(0, 0)] * vecs[i][0] + r[(0, 1)] * vecs[i][1] + r[(0, 2)] * vecs[i][2];
        out[i][1] = r[(1, 0)] * vecs[i][0] + r[(1, 1)] * vecs[i][1] + r[(1, 2)] * vecs[i][2];
        out[i][2] = r[(2, 0)] * vecs[i][0] + r[(2, 1)] * vecs[i][1] + r[(2, 2)] * vecs[i][2];
    }
}

// OPTIMIZATION: Pure-Rust stack-allocated SVD (via nalgebra). Replaces dynamic DMatrix matching.
fn find_rotation_matrix_and_det_inplace(
    image_vectors: &[[f64; 3]],
    catalog_vectors: &[[f64; 3]],
    len: usize,
) -> (Matrix3<f64>, f64) {
    let mut h = Matrix3::<f64>::zeros();
    // H = image_vectors.T * catalog_vectors
    for i in 0..len {
        for r in 0..3 {
            for c in 0..3 {
                h[(r, c)] += image_vectors[i][r] * catalog_vectors[i][c];
            }
        }
    }

    let svd = SVD::new(h, true, true);
    if let (Some(u), Some(vt)) = (svd.u, svd.v_t) {
        let rot = u * vt;
        (rot, rot.determinant())
    } else {
        (Matrix3::zeros(), -1.0)
    }
}

// OPTIMIZATION: Zero-allocation inner loop alternative to matching logic
fn find_centroid_matches_inplace(
    image_centroids: &[[f64; 2]],
    img_len: usize,
    catalog_centroids: &[[f64; 2]],
    cat_len: usize,
    r: f64,
) -> Vec<(usize, usize)> {
    let mut matches = Vec::new();
    let r_sq = r * r;
    for i in 0..img_len {
        for j in 0..cat_len {
            let dy = image_centroids[i][0] - catalog_centroids[j][0];
            let dx = image_centroids[i][1] - catalog_centroids[j][1];
            if (dy * dy + dx * dx) < r_sq {
                matches.push((i, j));
            }
        }
    }

    let mut unique_col1 = std::collections::BTreeMap::new();
    for (idx, &(_, j)) in matches.iter().enumerate() {
        unique_col1.entry(j).or_insert(idx);
    }
    let indices1: Vec<usize> = unique_col1.values().cloned().collect();
    let matches1: Vec<(usize, usize)> = indices1.into_iter().map(|idx| matches[idx]).collect();

    let mut unique_col0 = std::collections::BTreeMap::new();
    for (idx, &(i, _)) in matches1.iter().enumerate() {
        unique_col0.entry(i).or_insert(idx);
    }
    let indices0: Vec<usize> = unique_col0.values().cloned().collect();
    indices0.into_iter().map(|idx| matches1[idx]).collect()
}

// OPTIMIZATION: Replaces slow `itertools::multi_cartesian_product` with zero-allocation DFS
fn generate_pattern_keys(
    mins: &[usize],
    maxs: &[usize],
    targets: &[usize],
    current: &mut Vec<usize>,
    out: &mut Vec<(usize, Vec<usize>)>,
) {
    let depth = current.len();
    if depth == mins.len() {
        let mut dist = 0;
        for i in 0..depth {
            let diff = (current[i] as isize - targets[i] as isize).abs();
            dist += diff * diff;
        }
        out.push((dist as usize, current.clone()));
        return;
    }
    for val in mins[depth]..=maxs[depth] {
        current.push(val);
        generate_pattern_keys(mins, maxs, targets, current, out);
        current.pop();
    }
}

// Helper to build the solution
#[allow(clippy::too_many_arguments)]
fn verify_and_build_solution(
    scratch: &mut Scratchpads,
    star_kd_tree: &KdTree<f64, 3>,
    star_table_flat: &[CatalogStar],
    star_catalog_ids: &Option<Array1<u32>>,
    db_props: &HashMap<String, f64>,
    num_patterns: usize,
    rotation_matrix: &Matrix3<f64>,
    mut fov: f64,
    height: f64,
    width: f64,
    options: &SolveOptions,
    image_centroids: &[[f64; 2]],
    image_centroids_undist: &mut Vec<[f64; 2]>,
    num_extracted_stars: usize,
    match_threshold: f64,
    t0_solve: Instant,
) -> Option<Solution> {
    // Find all catalog stars inside the FOV diagonal
    let fov_diagonal_rad = fov * ((width * width + height * height).sqrt() / width);
    let image_center_vector = [
        rotation_matrix[(0, 0)],
        rotation_matrix[(0, 1)],
        rotation_matrix[(0, 2)],
    ];

    // Epsilon to capture borders safely in f64
    let max_dist_sq = distance_from_angle(fov_diagonal_rad / 2.0).powi(2) + 1e-8;
    let mut nearby_cat_stars_inds: Vec<usize> = star_kd_tree
        .within::<SquaredEuclidean>(&image_center_vector, max_dist_sq)
        .into_iter()
        .map(|n| n.item as usize)
        .collect();

    // Re-sort KDTree return list by index to prioritize brighter stars exactly like Python
    nearby_cat_stars_inds.sort_unstable();

    let num_nearby = nearby_cat_stars_inds.len();
    if num_nearby == 0 {
        return None;
    }

    if scratch.sp_nearby_cat_star_vectors.len() < num_nearby {
        let new_size = num_nearby.max(scratch.sp_nearby_cat_star_vectors.len() * 2);
        scratch
            .sp_nearby_cat_star_vectors
            .resize(new_size, [0.0; 3]);
        scratch
            .sp_nearby_cat_star_vectors_derot
            .resize(new_size, [0.0; 3]);
        scratch
            .sp_nearby_cat_star_centroids_all
            .resize(new_size, [0.0; 2]);
    }

    for (n_idx, &star_idx) in nearby_cat_stars_inds.iter().enumerate() {
        scratch.sp_nearby_cat_star_vectors[n_idx] = star_table_flat[star_idx].vec;
    }

    rotate_vectors_inplace(
        rotation_matrix,
        &scratch.sp_nearby_cat_star_vectors[..num_nearby],
        false,
        &mut scratch.sp_nearby_cat_star_vectors_derot[..num_nearby],
        num_nearby,
    );
    compute_centroids_inplace(
        &scratch.sp_nearby_cat_star_vectors_derot[..num_nearby],
        height,
        width,
        fov,
        &mut scratch.sp_nearby_cat_star_centroids_all[..num_nearby],
        &mut scratch.sp_kept,
        num_nearby,
    );

    let crop_len = scratch.sp_kept.len().min(2 * num_extracted_stars);

    if scratch.sp_valid_cat_centroids.len() < crop_len {
        let new_size = crop_len.max(scratch.sp_valid_cat_centroids.len() * 2);
        scratch.sp_valid_cat_centroids.resize(new_size, [0.0; 2]);
        scratch.sp_valid_cat_vectors.resize(new_size, [0.0; 3]);
    }
    scratch.sp_valid_cat_inds.clear();

    for (c_idx, &x) in scratch.sp_kept.iter().take(crop_len).enumerate() {
        scratch.sp_valid_cat_centroids[c_idx] = scratch.sp_nearby_cat_star_centroids_all[x];
        scratch.sp_valid_cat_vectors[c_idx] = scratch.sp_nearby_cat_star_vectors[x];
        scratch.sp_valid_cat_inds.push(nearby_cat_stars_inds[x]);
    }

    let matched_stars = find_centroid_matches_inplace(
        image_centroids_undist,
        num_extracted_stars,
        &scratch.sp_valid_cat_centroids,
        crop_len,
        width * options.match_radius,
    );

    // Probability Calculation
    let num_star_matches = matched_stars.len();
    let prob_single_star_mismatch = (crop_len as f64) * options.match_radius.powi(2);
    let p_raw = 1.0 - prob_single_star_mismatch;
    let k_raw = num_extracted_stars as i64 - (num_star_matches as i64 - 2);

    // Safe bounds bypass replicating scipy.stats.binom.cdf behavior
    let prob_mismatch = fast_binomial_cdf(k_raw, num_extracted_stars as u64, p_raw);
    if prob_mismatch >= match_threshold {
        return None;
    }

    // We passed all checks. Complete the final exact solution details
    let mut matched_img_cents = Vec::with_capacity(num_star_matches);
    let mut matched_cat_vecs = Vec::with_capacity(num_star_matches);
    for &(img_idx, cat_idx) in &matched_stars {
        matched_img_cents.push(image_centroids_undist[img_idx]);
        matched_cat_vecs.push(scratch.sp_valid_cat_vectors[cat_idx]);
    }

    let matched_img_vecs = compute_vectors_flat(&matched_img_cents, height, width, fov);
    let (precise_rotation_matrix, _) = find_rotation_matrix_and_det_inplace(
        &matched_img_vecs,
        &matched_cat_vecs,
        num_star_matches,
    );

    let mut k_final = options.distortion;
    if options.distortion.is_some() {
        // Refine fov & distortion using Least Squares System
        // A = [tangent, radius^3], b = [radius]
        // Note: To fully map lstsq in Rust precisely, build A and B for all matched_stars
        let mut a_na = DMatrix::<f64>::zeros(num_star_matches, 2);
        let mut b_na = DVector::<f64>::zeros(num_star_matches);
        let mut derotated_matched_cat = vec![[0.0; 3]; num_star_matches];
        rotate_vectors_inplace(
            &precise_rotation_matrix,
            &matched_cat_vecs,
            false,
            &mut derotated_matched_cat,
            num_star_matches,
        );

        for (m_idx, &(img_idx, _)) in matched_stars.iter().enumerate() {
            let r_cent = image_centroids[img_idx];
            let r_dist = ((r_cent[0] - height / 2.0).powi(2) + (r_cent[1] - width / 2.0).powi(2))
                .sqrt()
                / width
                * 2.0;
            let cat_derot = derotated_matched_cat[m_idx];
            let tangent = (cat_derot[1].powi(2) + cat_derot[2].powi(2)).sqrt() / cat_derot[0];
            a_na[(m_idx, 0)] = tangent;
            a_na[(m_idx, 1)] = r_dist.powi(3);
            b_na[m_idx] = r_dist;
        }

        // Pure-Rust SVD Pseudo-Inverse for Distortions
        let svd = SVD::new(a_na, true, true);
        if let Ok(pseudo_inv) = svd.pseudo_inverse(1e-7) {
            let sol = pseudo_inv * b_na;
            let f_val = sol[0] / (1.0 - sol[1]);
            k_final = Some(sol[1]);
            fov = 2.0 * (1.0 / f_val).atan();
            *image_centroids_undist = undistort_centroids(image_centroids, height, width, sol[1]);
            for (m_idx, &(img_idx, _)) in matched_stars.iter().enumerate() {
                matched_img_cents[m_idx] = image_centroids_undist[img_idx];
            }
        }
    }

    let final_match_vectors = compute_vectors_flat(&matched_img_cents, height, width, fov);
    let mut final_derotated = vec![[0.0; 3]; num_star_matches];
    rotate_vectors_inplace(
        &precise_rotation_matrix,
        &final_match_vectors,
        true,
        &mut final_derotated,
        num_star_matches,
    );

    let mut distances: Vec<f64> = (0..num_star_matches)
        .map(|m_idx| {
            let row_f = final_derotated[m_idx];
            let row_c = matched_cat_vecs[m_idx];
            ((row_f[0] - row_c[0]).powi(2)
                + (row_f[1] - row_c[1]).powi(2)
                + (row_f[2] - row_c[2]).powi(2))
            .sqrt()
        })
        .collect();
    distances.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let p90_idx = (0.9 * (distances.len() - 1) as f64) as usize;
    let p90_err_angle = angle_from_distance(distances[p90_idx]).to_degrees() * 3600.0;
    let max_err_angle = angle_from_distance(*distances.last().unwrap()).to_degrees() * 3600.0;

    let mut rms_sum = 0.0;
    for &d in &distances {
        let a = angle_from_distance(d);
        rms_sum += a * a;
    }
    let rms_err_angle = (rms_sum / distances.len() as f64).sqrt().to_degrees() * 3600.0;

    let ra = precise_rotation_matrix[(0, 1)]
        .atan2(precise_rotation_matrix[(0, 0)])
        .to_degrees()
        .rem_euclid(360.0);
    let dec = precise_rotation_matrix[(0, 2)]
        .atan2(
            (precise_rotation_matrix[(1, 2)].powi(2) + precise_rotation_matrix[(2, 2)].powi(2))
                .sqrt(),
        )
        .to_degrees();
    let roll = precise_rotation_matrix[(1, 2)]
        .atan2(precise_rotation_matrix[(2, 2)])
        .to_degrees()
        .rem_euclid(360.0);

    let mut precise_rot_arr2 = Array2::zeros((3, 3));
    for r in 0..3 {
        for c in 0..3 {
            precise_rot_arr2[[r, c]] = precise_rotation_matrix[(r, c)];
        }
    }

    let mut solution = Solution {
        ra: Some(ra),
        dec: Some(dec),
        roll: Some(roll),
        fov: Some(fov.to_degrees()),
        distortion: k_final,
        rmse: Some(rms_err_angle),
        p90e: Some(p90_err_angle),
        maxe: Some(max_err_angle),
        matches: Some(num_star_matches),
        prob: Some(prob_mismatch * (num_patterns as f64)),
        epoch_equinox: db_props.get("epoch_equinox").cloned(),
        epoch_proper_motion: db_props.get("epoch_proper_motion").cloned(),
        status: SolveStatus::MatchFound,
        t_solve_ms: t0_solve.elapsed().as_secs_f64() * 1000.0,
        ..Default::default()
    };

    if options.return_rotation_matrix {
        solution.rotation_matrix = Some(precise_rot_arr2);
    }
    if let Some(target_px) = &options.target_pixel {
        let mut px_flat = Vec::with_capacity(target_px.nrows());
        for row in 0..target_px.nrows() {
            px_flat.push([target_px[[row, 0]], target_px[[row, 1]]]);
        }
        if let Some(k) = k_final {
            px_flat = undistort_centroids(&px_flat, height, width, k);
        }
        let target_vector = compute_vectors_flat(&px_flat, height, width, fov);
        let mut rotated_target_vector = vec![[0.0; 3]; target_vector.len()];
        rotate_vectors_inplace(
            &precise_rotation_matrix,
            &target_vector,
            true,
            &mut rotated_target_vector,
            target_vector.len(),
        );
        let mut target_ra = Vec::new();
        let mut target_dec = Vec::new();
        for v in rotated_target_vector {
            target_ra.push(v[1].atan2(v[0]).to_degrees().rem_euclid(360.0));
            target_dec.push(90.0 - v[2].acos().to_degrees());
        }
        solution.target_ra = Some(target_ra);
        solution.target_dec = Some(target_dec);
    }
    if let Some(target_sky) = &options.target_sky_coord {
        let mut target_sky_vecs = Vec::with_capacity(target_sky.nrows());
        for row in 0..target_sky.nrows() {
            let ra_rad = target_sky[[row, 0]].to_radians();
            let dec_rad = target_sky[[row, 1]].to_radians();
            target_sky_vecs.push([
                ra_rad.cos() * dec_rad.cos(),
                ra_rad.sin() * dec_rad.cos(),
                dec_rad.sin(),
            ]);
        }
        let mut target_sky_vecs_derot = vec![[0.0; 3]; target_sky_vecs.len()];
        rotate_vectors_inplace(
            &precise_rotation_matrix,
            &target_sky_vecs,
            false,
            &mut target_sky_vecs_derot,
            target_sky_vecs.len(),
        );
        let mut target_centroids = vec![[0.0; 2]; target_sky_vecs.len()];
        let mut kept_sky = Vec::new();
        compute_centroids_inplace(
            &target_sky_vecs_derot,
            height,
            width,
            fov,
            &mut target_centroids,
            &mut kept_sky,
            target_sky_vecs.len(),
        );
        if let Some(k) = k_final {
            for &k_idx in &kept_sky {
                let distorted =
                    distort_centroids(&[target_centroids[k_idx]], height, width, k, 1e-6, 30);
                target_centroids[k_idx] = distorted[0];
            }
        }
        let mut target_y = vec![None; target_sky.nrows()];
        let mut target_x = vec![None; target_sky.nrows()];
        for &k_idx in &kept_sky {
            target_y[k_idx] = Some(target_centroids[k_idx][0]);
            target_x[k_idx] = Some(target_centroids[k_idx][1]);
        }
        solution.target_y = Some(target_y);
        solution.target_x = Some(target_x);
    }
    if options.return_matches {
        let mut m_cents = Vec::new();
        let mut m_stars = Vec::new();
        let mut m_ids = Vec::new();
        for &(img_idx, cat_idx) in &matched_stars {
            m_cents.push(image_centroids_undist[img_idx]);
            let star_idx = scratch.sp_valid_cat_inds[cat_idx];
            let star = &star_table_flat[star_idx];
            m_stars.push([star.ra.to_degrees(), star.dec.to_degrees(), star.mag]);
            if let Some(ids) = star_catalog_ids {
                m_ids.push(ids[star_idx]);
            }
        }
        solution.matched_centroids = Some(m_cents);
        solution.matched_stars = Some(m_stars);
        if !m_ids.is_empty() {
            solution.matched_cat_id = Some(m_ids);
        }
    }
    if options.return_catalog {
        let mut cat_stars = Vec::new();
        for c_idx in 0..crop_len {
            let star_idx = scratch.sp_valid_cat_inds[c_idx];
            let star = &star_table_flat[star_idx];
            cat_stars.push((
                star.ra.to_degrees(),
                star.dec.to_degrees(),
                star.mag,
                scratch.sp_valid_cat_centroids[c_idx][0],
                scratch.sp_valid_cat_centroids[c_idx][1],
            ));
        }
        solution.catalog_stars = Some(cat_stars);
    }

    Some(solution)
}

// --- Ported Utility Functions ---

fn separation_for_density(fov: f64, stars_per_fov: f64) -> f64 {
    0.6 * fov / stars_per_fov.sqrt()
}

// Fallback iterator for non-standard p_size configurations
fn breadth_first_combinations(sequence: &[usize], r: usize) -> Vec<Vec<usize>> {
    let mut results = Vec::new();
    if r == 1 {
        for &item in sequence {
            results.push(vec![item]);
        }
        return results;
    }
    let mut index = r - 1;
    while index < sequence.len() {
        let right_most_elt = sequence[index];
        let prefixes = breadth_first_combinations(&sequence[..index], r - 1);
        for mut prefix in prefixes {
            prefix.push(right_most_elt);
            results.push(prefix);
        }
        index += 1;
    }
    results
}

// --- Scratchpad for Zero-Allocation Combinatorics ---
#[derive(Default)]
pub struct Scratchpads {
    // Fast path structs (p_size == 4)
    pub sp_pattern_key_list: Vec<(usize, [usize; 5])>,

    // Core matching scratchpads
    pub sp_cat_edges_list: Vec<Vec<f64>>,
    pub sp_cat_vectors_list: Vec<Vec<[f64; 3]>>,
    pub sp_p_cents: Vec<[f64; 2]>,
    pub sp_p_vecs: Vec<[f64; 3]>,
    pub sp_image_pattern_vectors_sorted: Vec<[f64; 3]>,
    pub sp_radii_scratch: Vec<(f64, usize)>,
    pub sp_catalog_pattern_vectors_sorted: Vec<[f64; 3]>,
    pub sp_nearby_cat_star_vectors: Vec<[f64; 3]>,
    pub sp_nearby_cat_star_vectors_derot: Vec<[f64; 3]>,
    pub sp_nearby_cat_star_centroids_all: Vec<[f64; 2]>,
    pub sp_kept: Vec<usize>,
    pub sp_valid_cat_centroids: Vec<[f64; 2]>,
    pub sp_valid_cat_vectors: Vec<[f64; 3]>,
    pub sp_valid_cat_inds: Vec<usize>,
    pub sp_hash_match_inds: Vec<usize>,

    // Fallback path structs (p_size != 4)
    pub sp_pattern_vecs: Vec<[f64; 3]>,
    pub sp_dists: Vec<f64>,
    pub sp_edge_angles: Vec<f64>,
    pub sp_image_pattern: Vec<f64>,
    pub sp_pattern_key_space_min: Vec<usize>,
    pub sp_pattern_key_space_max: Vec<usize>,
    pub sp_image_pattern_key: Vec<usize>,
    pub sp_pattern_key_current: Vec<usize>,
    pub sp_pattern_key_list_fallback: Vec<(usize, Vec<usize>)>,
}

impl Scratchpads {
    pub fn new(p_size: usize) -> Self {
        let max_size = p_size.max(6);
        Self {
            sp_pattern_key_list: Vec::with_capacity(512),
            sp_cat_edges_list: Vec::with_capacity(32),
            sp_cat_vectors_list: Vec::with_capacity(32),

            sp_p_cents: vec![[0.0; 2]; max_size],
            sp_p_vecs: vec![[0.0; 3]; max_size],
            sp_image_pattern_vectors_sorted: vec![[0.0; 3]; max_size],
            sp_radii_scratch: Vec::with_capacity(max_size),
            sp_catalog_pattern_vectors_sorted: vec![[0.0; 3]; max_size],

            sp_nearby_cat_star_vectors: Vec::with_capacity(256),
            sp_nearby_cat_star_vectors_derot: Vec::with_capacity(256),
            sp_nearby_cat_star_centroids_all: Vec::with_capacity(256),
            sp_kept: Vec::with_capacity(256),

            sp_valid_cat_centroids: Vec::with_capacity(256),
            sp_valid_cat_vectors: Vec::with_capacity(256),
            sp_valid_cat_inds: Vec::with_capacity(256),
            sp_hash_match_inds: Vec::with_capacity(64),

            // Fallback path allocations
            sp_pattern_vecs: vec![[0.0; 3]; max_size],
            sp_dists: Vec::with_capacity(16),
            sp_edge_angles: Vec::with_capacity(16),
            sp_image_pattern: Vec::with_capacity(16),
            sp_pattern_key_space_min: Vec::with_capacity(16),
            sp_pattern_key_space_max: Vec::with_capacity(16),
            sp_image_pattern_key: Vec::with_capacity(16),
            sp_pattern_key_current: Vec::with_capacity(16),
            sp_pattern_key_list_fallback: Vec::with_capacity(512),
        }
    }
}

// --- Main Engine ---

pub struct Tetra3 {
    // OPTIMIZATION: Highly optimized cache-aligned struct lists
    pub star_table_flat: Vec<CatalogStar>,
    pub pattern_catalog_flat: Vec<usize>,
    pub star_kd_tree: KdTree<f64, 3>,

    pub pattern_largest_edge: Option<Array1<f32>>,
    pub pattern_key_hashes: Option<Array1<u16>>,
    pub star_catalog_ids: Option<Array1<u32>>,
    pub db_props: HashMap<String, f64>,
    pub num_patterns: usize,
    pub linear_probe: bool,
    pub scratch: Scratchpads, // OPTIMIZATION: Instance-level persistent memory context
    cancelled: AtomicBool,
}

impl Tetra3 {
    pub fn load_database(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut archive = ZipArchive::new(file)?;

        let read_star_table = |arc: &mut ZipArchive<File>,
                               name: &str|
         -> Result<Array2<f64>, Box<dyn std::error::Error>> {
            let mut zf = arc.by_name(name)?;
            let mut buf = Vec::new();
            zf.read_to_end(&mut buf)?;
            let mut cursor = Cursor::new(&buf);
            let npy = NpyFile::new(&mut cursor)?;
            let shape = npy.shape().to_vec();
            let mut cursor = Cursor::new(&buf);
            let npy2 = NpyFile::new(&mut cursor)?;
            if let Ok(data) = npy2.into_vec::<f64>() {
                return Ok(Array2::from_shape_vec(
                    (shape[0] as usize, shape[1] as usize),
                    data,
                )?);
            }
            let mut cursor = Cursor::new(&buf);
            let npy3 = NpyFile::new(&mut cursor)?;
            let data_f32: Vec<f32> = npy3.into_vec()?;
            let data: Vec<f64> = data_f32.into_iter().map(|v| v as f64).collect();
            Ok(Array2::from_shape_vec(
                (shape[0] as usize, shape[1] as usize),
                data,
            )?)
        };

        let read_pattern_catalog = |arc: &mut ZipArchive<File>,
                                    name: &str|
         -> Result<Array2<usize>, Box<dyn std::error::Error>> {
            let mut zf = arc.by_name(name)?;
            let mut buf = Vec::new();
            zf.read_to_end(&mut buf)?;
            let mut cursor = Cursor::new(&buf);
            let npy = NpyFile::new(&mut cursor)?;
            let shape = npy.shape().to_vec();
            let mut cursor = Cursor::new(&buf);
            let npy2 = NpyFile::new(&mut cursor)?;
            if let Ok(data_u16) = npy2.into_vec::<u16>() {
                let data: Vec<usize> = data_u16.into_iter().map(|v| v as usize).collect();
                return Ok(Array2::from_shape_vec(
                    (shape[0] as usize, shape[1] as usize),
                    data,
                )?);
            }
            let mut cursor = Cursor::new(&buf);
            let npy3 = NpyFile::new(&mut cursor)?;
            let data_u32: Vec<u32> = npy3.into_vec()?;
            let data: Vec<usize> = data_u32.into_iter().map(|v| v as usize).collect();
            Ok(Array2::from_shape_vec(
                (shape[0] as usize, shape[1] as usize),
                data,
            )?)
        };

        let read_1d_f32 = |arc: &mut ZipArchive<File>, name: &str| -> Option<Array1<f32>> {
            arc.by_name(name).ok().and_then(|mut zf| {
                let mut buf = Vec::new();
                zf.read_to_end(&mut buf).ok()?;
                let mut cursor = Cursor::new(&buf);
                let npy = NpyFile::new(&mut cursor).ok()?;
                Some(Array1::from_vec(npy.into_vec().ok()?))
            })
        };

        let read_1d_u16 = |arc: &mut ZipArchive<File>, name: &str| -> Option<Array1<u16>> {
            arc.by_name(name).ok().and_then(|mut zf| {
                let mut buf = Vec::new();
                zf.read_to_end(&mut buf).ok()?;
                let mut cursor = Cursor::new(&buf);
                let npy = NpyFile::new(&mut cursor).ok()?;
                Some(Array1::from_vec(npy.into_vec().ok()?))
            })
        };

        let read_1d_u32 = |arc: &mut ZipArchive<File>, name: &str| -> Option<Array1<u32>> {
            arc.by_name(name).ok().and_then(|mut zf| {
                let mut buf = Vec::new();
                zf.read_to_end(&mut buf).ok()?;
                let mut cursor = Cursor::new(&buf);
                let npy = NpyFile::new(&mut cursor).ok()?;
                Some(Array1::from_vec(npy.into_vec().ok()?))
            })
        };

        let pattern_catalog_arr = read_pattern_catalog(&mut archive, "pattern_catalog.npy")?;
        let star_table_arr = read_star_table(&mut archive, "star_table.npy")?;
        let pattern_largest_edge = read_1d_f32(&mut archive, "pattern_largest_edge.npy");
        let pattern_key_hashes = read_1d_u16(&mut archive, "pattern_key_hashes.npy");
        let star_catalog_ids = read_1d_u32(&mut archive, "star_catalog_IDs.npy");

        // OPTIMIZATION: Convert massive Array2 allocations to deeply mapped fast internal native slices
        let mut star_table_flat = Vec::with_capacity(star_table_arr.nrows());
        for i in 0..star_table_arr.nrows() {
            star_table_flat.push(CatalogStar {
                ra: star_table_arr[[i, 0]],
                dec: star_table_arr[[i, 1]],
                vec: [
                    star_table_arr[[i, 2]],
                    star_table_arr[[i, 3]],
                    star_table_arr[[i, 4]],
                ],
                mag: star_table_arr[[i, 5]],
            });
        }

        let mut pattern_catalog_flat = Vec::with_capacity(pattern_catalog_arr.len());
        for &val in pattern_catalog_arr.iter() {
            pattern_catalog_flat.push(val);
        }

        let mut star_kd_tree = KdTree::new();
        for (i, star) in star_table_flat.iter().enumerate() {
            star_kd_tree.add(&star.vec, i as u64);
        }

        let mut num_patterns = pattern_catalog_arr.nrows() / 2;
        let mut db_props = HashMap::new();
        let mut linear_probe = false;

        db_props.insert("pattern_size".to_string(), 4.0);
        db_props.insert("pattern_bins".to_string(), 50.0);
        db_props.insert("pattern_max_error".to_string(), 0.002);
        db_props.insert("verification_stars_per_fov".to_string(), 10.0);
        db_props.insert("max_fov".to_string(), 20.0);
        db_props.insert("min_fov".to_string(), 20.0);
        db_props.insert("epoch_equinox".to_string(), 2000.0);
        db_props.insert("presort_patterns".to_string(), 0.0);

        if let Ok(mut zf) = archive.by_name("props_packed.npy") {
            let mut buf = Vec::new();
            if zf.read_to_end(&mut buf).is_ok() {
                let mut cursor = Cursor::new(&buf);
                if NpyFile::new(&mut cursor).is_ok() {
                    let mut data = vec![0u8; 828];
                    if cursor.read_exact(&mut data).is_ok() {
                        let mut hash_type = String::new();
                        for i in 0..64 {
                            let offset = 256 + (i * 4);
                            let c = data[offset];
                            if c == 0 {
                                break;
                            }
                            hash_type.push(c as char);
                        }
                        if hash_type.trim() == "linear_probe" {
                            linear_probe = true;
                        }

                        let p_size = u16::from_le_bytes([data[512], data[513]]);
                        db_props.insert("pattern_size".to_string(), p_size as f64);
                        let p_bins = u16::from_le_bytes([data[514], data[515]]);
                        db_props.insert("pattern_bins".to_string(), p_bins as f64);
                        let p_max_err =
                            f32::from_le_bytes([data[516], data[517], data[518], data[519]]);
                        db_props.insert("pattern_max_error".to_string(), p_max_err as f64);
                        let max_fov =
                            f32::from_le_bytes([data[520], data[521], data[522], data[523]]);
                        db_props.insert("max_fov".to_string(), max_fov as f64);
                        let min_fov =
                            f32::from_le_bytes([data[524], data[525], data[526], data[527]]);
                        db_props.insert("min_fov".to_string(), min_fov as f64);
                        let eq = u16::from_le_bytes([data[784], data[785]]);
                        db_props.insert("epoch_equinox".to_string(), eq as f64);
                        let pm = f32::from_le_bytes([data[786], data[787], data[788], data[789]]);
                        db_props.insert("epoch_proper_motion".to_string(), pm as f64);
                        let vs = u16::from_le_bytes([data[800], data[801]]);
                        db_props.insert("verification_stars_per_fov".to_string(), vs as f64);
                        let presort = data[823] != 0;
                        db_props.insert(
                            "presort_patterns".to_string(),
                            if presort { 1.0 } else { 0.0 },
                        );
                        num_patterns =
                            u32::from_le_bytes([data[824], data[825], data[826], data[827]])
                                as usize;
                    }
                }
            }
        }

        let p_size = *db_props.get("pattern_size").unwrap_or(&4.0) as usize;

        Ok(Tetra3 {
            star_table_flat,
            pattern_catalog_flat,
            star_kd_tree,
            pattern_largest_edge,
            pattern_key_hashes,
            star_catalog_ids,
            db_props,
            num_patterns,
            linear_probe,
            scratch: Scratchpads::new(p_size),
            cancelled: AtomicBool::new(false),
        })
    }

    pub fn cancel_solve(&mut self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    #[inline(always)]
    fn compute_pattern_key_hash_4(pattern_key: &[usize; 5], bin_factor: usize) -> u64 {
        let mut hash: u64 = 0;
        let mut multiplier: u64 = 1;
        for &k in pattern_key {
            hash += (k as u64) * multiplier;
            multiplier *= bin_factor as u64;
        }
        hash
    }

    #[inline(always)]
    fn compute_pattern_key_hash(pattern_key: &[usize], bin_factor: usize) -> u64 {
        let mut hash: u64 = 0;
        let mut multiplier: u64 = 1;
        for &k in pattern_key {
            hash += (k as u64) * multiplier;
            multiplier *= bin_factor as u64;
        }
        hash
    }

    #[inline(always)]
    fn pattern_key_hash_to_index(hash: u64, max_index: u64, linear_probe: bool) -> u64 {
        if linear_probe {
            hash % max_index
        } else {
            hash.wrapping_mul(MAGIC_RAND) % max_index
        }
    }

    fn get_table_indices_from_hash_inplace(
        pattern_catalog_flat: &[usize],
        p_size: usize,
        hash_index: u64,
        linear_probe: bool,
        out_found: &mut Vec<usize>,
    ) {
        out_found.clear();
        let max_ind = (pattern_catalog_flat.len() / p_size) as u64;
        for c in 0.. {
            let i = if linear_probe {
                (hash_index + c) % max_ind
            } else {
                (hash_index + c * c) % max_ind
            };
            let row_start = (i as usize) * p_size;

            let mut is_empty = true;
            for j in 0..p_size {
                if pattern_catalog_flat[row_start + j] != 0 {
                    is_empty = false;
                    break;
                }
            }
            if is_empty {
                break;
            }
            out_found.push(i as usize);
        }
    }

    fn get_all_patterns_for_index_inplace(
        pattern_key_hash: u64,
        hash_index: u64,
        image_pattern_largest_edge: f64,
        fov_estimate: Option<f64>,
        fov_max_error: Option<f64>,
        pattern_catalog_flat: &[usize],
        p_size: usize,
        pattern_key_hashes: &Option<Array1<u16>>,
        pattern_largest_edge: &Option<Array1<f32>>,
        star_table_flat: &[CatalogStar],
        linear_probe: bool,
        sp_hash_match_inds: &mut Vec<usize>,
        out_edges: &mut Vec<Vec<f64>>,
        out_vectors: &mut Vec<Vec<[f64; 3]>>,
    ) {
        Self::get_table_indices_from_hash_inplace(
            pattern_catalog_flat,
            p_size,
            hash_index,
            linear_probe,
            sp_hash_match_inds,
        );
        if sp_hash_match_inds.is_empty() {
            out_edges.clear();
            out_vectors.clear();
            return;
        }

        if let Some(hashes) = pattern_key_hashes {
            let key_hash16 = (pattern_key_hash & 0xffff) as u16;
            sp_hash_match_inds.retain(|&idx| hashes[idx] == key_hash16);
        }

        if let (Some(largest_edges), Some(f_est), Some(f_err)) =
            (pattern_largest_edge, fov_estimate, fov_max_error)
        {
            sp_hash_match_inds.retain(|&idx| {
                let cat_largest_edge = largest_edges[idx] as f64;
                let fov2 = cat_largest_edge / image_pattern_largest_edge * f_est / 1000.0;
                (fov2 - f_est).abs() < f_err
            });
        }

        let num_matches = sp_hash_match_inds.len();
        while out_edges.len() < num_matches {
            out_edges.push(Vec::with_capacity(16));
        }
        while out_vectors.len() < num_matches {
            out_vectors.push(vec![[0.0; 3]; p_size]);
        }
        out_edges.truncate(num_matches);
        out_vectors.truncate(num_matches);

        for (out_idx, &idx) in sp_hash_match_inds.iter().enumerate() {
            let row_start = idx * p_size;
            let vecs = &mut out_vectors[out_idx];
            for i in 0..p_size {
                let star_id = pattern_catalog_flat[row_start + i];
                vecs[i] = star_table_flat[star_id].vec;
            }

            let edges_vec = &mut out_edges[out_idx];
            edges_vec.clear();
            for i in 0..p_size {
                for j in (i + 1)..p_size {
                    let d0 = vecs[i][0] - vecs[j][0];
                    let d1 = vecs[i][1] - vecs[j][1];
                    let d2 = vecs[i][2] - vecs[j][2];
                    edges_vec.push(angle_from_distance((d0 * d0 + d1 * d1 + d2 * d2).sqrt()));
                }
            }
            edges_vec.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        }
    }

    pub fn solve_from_centroids(
        &mut self,
        star_centroids: &Array2<f64>,
        size: (f64, f64),
        options: SolveOptions,
    ) -> Solution {
        let t0_solve = Instant::now();
        let (height, width) = size;

        self.cancelled.store(false, Ordering::Relaxed);

        let fov_initial = options
            .fov_estimate
            .map(|f| f.to_radians())
            .unwrap_or_else(|| {
                let max_f = self.db_props.get("max_fov").unwrap_or(&20.0);
                let min_f = self.db_props.get("min_fov").unwrap_or(&10.0);
                ((max_f + min_f) / 2.0).to_radians()
            });

        let p_size = *self.db_props.get("pattern_size").unwrap_or(&4.0) as usize;
        let p_bins = *self.db_props.get("pattern_bins").unwrap_or(&50.0) as usize;
        let verification_stars = *self
            .db_props
            .get("verification_stars_per_fov")
            .unwrap_or(&10.0) as usize;
        let p_max_err = options
            .match_max_error
            .max(*self.db_props.get("pattern_max_error").unwrap_or(&0.002));
        let match_threshold = options.match_threshold / (self.num_patterns as f64);
        let presorted = *self.db_props.get("presort_patterns").unwrap_or(&0.0) == 1.0;

        let num_centroids_raw = star_centroids.nrows();
        if num_centroids_raw < p_size {
            return Solution {
                status: SolveStatus::TooFew,
                t_solve_ms: t0_solve.elapsed().as_secs_f64() * 1000.0,
                ..Default::default()
            };
        }

        // Thinning Strategy
        let pattern_stars_separation_pixels =
            width * separation_for_density(fov_initial, verification_stars as f64) / fov_initial;
        let mut keep_for_patterns = vec![false; num_centroids_raw];

        for i in 0..num_centroids_raw {
            let mut occupied = false;
            let c_i = star_centroids.row(i);
            for j in 0..i {
                if keep_for_patterns[j] {
                    let c_j = star_centroids.row(j);
                    if ((c_i[0] - c_j[0]).powi(2) + (c_i[1] - c_j[1]).powi(2)).sqrt()
                        < pattern_stars_separation_pixels
                    {
                        occupied = true;
                        break;
                    }
                }
            }
            if !occupied {
                keep_for_patterns[i] = true;
            }
        }

        let mut pattern_centroids_inds: Vec<usize> = keep_for_patterns
            .into_iter()
            .enumerate()
            .filter_map(|(i, keep)| if keep { Some(i) } else { None })
            .collect();

        let mut num_extracted_stars = num_centroids_raw;
        if num_centroids_raw > verification_stars {
            num_extracted_stars = verification_stars;
            pattern_centroids_inds.retain(|&i| i < num_extracted_stars);
        }

        // Maintain the original full set of image_centroids for the final matrix building
        let mut image_centroids = Vec::with_capacity(num_extracted_stars);
        for i in 0..num_extracted_stars {
            image_centroids.push([star_centroids[[i, 0]], star_centroids[[i, 1]]]);
        }

        let mut image_centroids_undist = if let Some(k) = options.distortion {
            undistort_centroids(&image_centroids, height, width, k)
        } else {
            image_centroids.clone()
        };

        let image_centroids_vectors =
            compute_vectors_flat(&image_centroids_undist, height, width, fov_initial);

        // OPTIMIZATION: Precompute pairwise distance angles exactly ONCE.
        // Drops 6 sqrt and 6 asin operations per iteration inside the hot combinatorics loop.
        let num_vecs = image_centroids_vectors.len();
        let mut precomputed_angles = vec![0.0; num_vecs * num_vecs];
        for i in 0..num_vecs {
            for j in (i + 1)..num_vecs {
                let v_i = image_centroids_vectors[i];
                let v_j = image_centroids_vectors[j];
                let dist = ((v_i[0] - v_j[0]).powi(2)
                    + (v_i[1] - v_j[1]).powi(2)
                    + (v_i[2] - v_j[2]).powi(2))
                .sqrt();
                let ang = angle_from_distance(dist);
                precomputed_angles[i * num_vecs + j] = ang;
                precomputed_angles[j * num_vecs + i] = ang;
            }
        }

        let scratch = &mut self.scratch;
        let star_kd_tree = &self.star_kd_tree;
        let star_table_flat = &self.star_table_flat;
        let pattern_catalog_flat = &self.pattern_catalog_flat;
        let pattern_key_hashes = &self.pattern_key_hashes;
        let pattern_largest_edge = &self.pattern_largest_edge;
        let linear_probe = self.linear_probe;

        let n_inds = pattern_centroids_inds.len();

        // -------------------------------------------------------------
        // HOT PATH: p_size == 4
        // Allocation-Free Native Iteration mirroring breadth-first order
        // -------------------------------------------------------------
        if p_size == 4 && n_inds >= 4 {
            for l in 3..n_inds {
                // OPTIMIZATION: Moving timeout and cancellation checks here.
                // Checking Instant::now() is a syscall. Moved out of the absolute deepest loop.
                if let Some(timeout) = options.solve_timeout_ms {
                    if t0_solve.elapsed().as_secs_f64() * 1000.0 > timeout {
                        return Solution {
                            status: SolveStatus::Timeout,
                            t_solve_ms: t0_solve.elapsed().as_secs_f64() * 1000.0,
                            ..Default::default()
                        };
                    }
                }
                if self.cancelled.load(Ordering::Relaxed) {
                    return Solution {
                        status: SolveStatus::Cancelled,
                        t_solve_ms: t0_solve.elapsed().as_secs_f64() * 1000.0,
                        ..Default::default()
                    };
                }

                for k in 2..l {
                    for j in 1..k {
                        for i in 0..j {
                            let p_i = pattern_centroids_inds[i];
                            let p_j = pattern_centroids_inds[j];
                            let p_k = pattern_centroids_inds[k];
                            let p_l = pattern_centroids_inds[l];

                            // Fast direct memory lookups for pairwise distance angle metrics
                            let mut edges = [
                                precomputed_angles[p_i * num_vecs + p_j],
                                precomputed_angles[p_i * num_vecs + p_k],
                                precomputed_angles[p_i * num_vecs + p_l],
                                precomputed_angles[p_j * num_vecs + p_k],
                                precomputed_angles[p_j * num_vecs + p_l],
                                precomputed_angles[p_k * num_vecs + p_l],
                            ];

                            // Calculate edge angles and sort
                            edges.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                            let image_pattern_largest_edge = edges[5];

                            // Min/max edge ration bounds
                            let mut key_space_min = [0; 5];
                            let mut key_space_max = [0; 5];
                            let mut target_keys = [0isize; 5];

                            for x in 0..5 {
                                let ratio = edges[x] / image_pattern_largest_edge;
                                key_space_min[x] =
                                    ((ratio - p_max_err).max(0.0) * p_bins as f64) as usize;
                                key_space_max[x] =
                                    ((ratio + p_max_err).min(1.0) * p_bins as f64) as usize;
                                target_keys[x] = (ratio * p_bins as f64) as isize;
                            }

                            // Generate search space combinations via zero-allocation DFS (replaces Cartesian product)
                            scratch.sp_pattern_key_list.clear();
                            for k0 in key_space_min[0]..=key_space_max[0] {
                                for k1 in key_space_min[1]..=key_space_max[1] {
                                    for k2 in key_space_min[2]..=key_space_max[2] {
                                        for k3 in key_space_min[3]..=key_space_max[3] {
                                            for k4 in key_space_min[4]..=key_space_max[4] {
                                                let diff0 = k0 as isize - target_keys[0];
                                                let diff1 = k1 as isize - target_keys[1];
                                                let diff2 = k2 as isize - target_keys[2];
                                                let diff3 = k3 as isize - target_keys[3];
                                                let diff4 = k4 as isize - target_keys[4];
                                                let dist = diff0 * diff0
                                                    + diff1 * diff1
                                                    + diff2 * diff2
                                                    + diff3 * diff3
                                                    + diff4 * diff4;
                                                scratch
                                                    .sp_pattern_key_list
                                                    .push((dist as usize, [k0, k1, k2, k3, k4]));
                                            }
                                        }
                                    }
                                }
                            }

                            let mut image_pattern_largest_distance = None;

                            for key_idx in 0..scratch.sp_pattern_key_list.len() {
                                let pattern_key = scratch.sp_pattern_key_list[key_idx].1;
                                let pattern_key_hash =
                                    Self::compute_pattern_key_hash_4(&pattern_key, p_bins);
                                let hash_index = Self::pattern_key_hash_to_index(
                                    pattern_key_hash,
                                    (pattern_catalog_flat.len() / p_size) as u64,
                                    linear_probe,
                                );

                                Self::get_all_patterns_for_index_inplace(
                                    pattern_key_hash,
                                    hash_index,
                                    image_pattern_largest_edge,
                                    options.fov_estimate.map(|x| x.to_radians()),
                                    options.fov_max_error.map(|x| x.to_radians()),
                                    pattern_catalog_flat,
                                    p_size,
                                    pattern_key_hashes,
                                    pattern_largest_edge,
                                    star_table_flat,
                                    linear_probe,
                                    &mut scratch.sp_hash_match_inds,
                                    &mut scratch.sp_cat_edges_list,
                                    &mut scratch.sp_cat_vectors_list,
                                );

                                for cat_idx in 0..scratch.sp_cat_edges_list.len() {
                                    let catalog_largest_edge =
                                        *scratch.sp_cat_edges_list[cat_idx].last().unwrap();
                                    let mut valid = true;
                                    for x in 0..5 {
                                        let cat_ratio = scratch.sp_cat_edges_list[cat_idx][x]
                                            / catalog_largest_edge;
                                        let img_ratio = edges[x] / image_pattern_largest_edge;
                                        if cat_ratio < img_ratio - p_max_err
                                            || cat_ratio > img_ratio + p_max_err
                                        {
                                            valid = false;
                                            break;
                                        }
                                    }
                                    if !valid {
                                        continue;
                                    }

                                    // We have a matched pattern! Calculate refined FOV
                                    let fov;
                                    if options.fov_estimate.is_some() {
                                        fov = catalog_largest_edge / image_pattern_largest_edge
                                            * fov_initial;
                                    } else {
                                        if image_pattern_largest_distance.is_none() {
                                            let pts = [
                                                image_centroids_undist[p_i],
                                                image_centroids_undist[p_j],
                                                image_centroids_undist[p_k],
                                                image_centroids_undist[p_l],
                                            ];
                                            let mut max_dist = 0.0;
                                            for r in 0..4 {
                                                for c in (r + 1)..4 {
                                                    let d = ((pts[r][0] - pts[c][0]).powi(2)
                                                        + (pts[r][1] - pts[c][1]).powi(2))
                                                    .sqrt();
                                                    if d > max_dist {
                                                        max_dist = d;
                                                    }
                                                }
                                            }
                                            image_pattern_largest_distance = Some(max_dist);
                                        }
                                        let f = image_pattern_largest_distance.unwrap()
                                            / 2.0
                                            / (catalog_largest_edge / 2.0).tan();
                                        fov = 2.0 * (width / 2.0 / f).atan();
                                    }

                                    // Re-calculate vectors uniquely sorted by radius to centroid
                                    let pts = [
                                        image_centroids_undist[p_i],
                                        image_centroids_undist[p_j],
                                        image_centroids_undist[p_k],
                                        image_centroids_undist[p_l],
                                    ];
                                    compute_vectors_inplace(
                                        &pts,
                                        height,
                                        width,
                                        fov,
                                        &mut scratch.sp_p_vecs,
                                        4,
                                    );
                                    sort_vectors_by_radius_inplace(
                                        &scratch.sp_p_vecs,
                                        &mut scratch.sp_image_pattern_vectors_sorted,
                                        &mut scratch.sp_radii_scratch,
                                        4,
                                    );

                                    if presorted {
                                        scratch.sp_catalog_pattern_vectors_sorted[..4]
                                            .copy_from_slice(
                                                &scratch.sp_cat_vectors_list[cat_idx][..4],
                                            );
                                    } else {
                                        sort_vectors_by_radius_inplace(
                                            &scratch.sp_cat_vectors_list[cat_idx],
                                            &mut scratch.sp_catalog_pattern_vectors_sorted,
                                            &mut scratch.sp_radii_scratch,
                                            4,
                                        );
                                    };

                                    let (rotation_matrix, det) =
                                        find_rotation_matrix_and_det_inplace(
                                            &scratch.sp_image_pattern_vectors_sorted,
                                            &scratch.sp_catalog_pattern_vectors_sorted,
                                            4,
                                        );
                                    if det < 0.0 {
                                        continue;
                                    }

                                    if let Some(solution) = verify_and_build_solution(
                                        scratch,
                                        star_kd_tree,
                                        star_table_flat,
                                        &self.star_catalog_ids,
                                        &self.db_props,
                                        self.num_patterns,
                                        &rotation_matrix,
                                        fov,
                                        height,
                                        width,
                                        &options,
                                        &image_centroids,
                                        &mut image_centroids_undist,
                                        num_extracted_stars,
                                        match_threshold,
                                        t0_solve,
                                    ) {
                                        return solution;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // -------------------------------------------------------------
        // FALLBACK PATH: Non-standard p_size (!= 4)
        // -------------------------------------------------------------
        else {
            for image_pattern_indices in breadth_first_combinations(&pattern_centroids_inds, p_size)
            {
                if let Some(timeout) = options.solve_timeout_ms {
                    if t0_solve.elapsed().as_secs_f64() * 1000.0 > timeout {
                        return Solution {
                            status: SolveStatus::Timeout,
                            t_solve_ms: t0_solve.elapsed().as_secs_f64() * 1000.0,
                            ..Default::default()
                        };
                    }
                }
                if self.cancelled.load(Ordering::Relaxed) {
                    return Solution {
                        status: SolveStatus::Cancelled,
                        t_solve_ms: t0_solve.elapsed().as_secs_f64() * 1000.0,
                        ..Default::default()
                    };
                }

                for (idx, &i) in image_pattern_indices.iter().enumerate() {
                    scratch.sp_pattern_vecs[idx] = image_centroids_vectors[i];
                }

                scratch.sp_dists.clear();
                for i in 0..p_size {
                    for j in (i + 1)..p_size {
                        let d0 = scratch.sp_pattern_vecs[i][0] - scratch.sp_pattern_vecs[j][0];
                        let d1 = scratch.sp_pattern_vecs[i][1] - scratch.sp_pattern_vecs[j][1];
                        let d2 = scratch.sp_pattern_vecs[i][2] - scratch.sp_pattern_vecs[j][2];
                        scratch.sp_dists.push((d0 * d0 + d1 * d1 + d2 * d2).sqrt());
                    }
                }

                // Calculate edge angles and sort
                scratch.sp_edge_angles.clear();
                for &d in &scratch.sp_dists {
                    scratch.sp_edge_angles.push(angle_from_distance(d));
                }
                scratch
                    .sp_edge_angles
                    .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

                let image_pattern_largest_edge = *scratch.sp_edge_angles.last().unwrap();
                scratch.sp_image_pattern.clear();
                for i in 0..(scratch.sp_edge_angles.len() - 1) {
                    scratch
                        .sp_image_pattern
                        .push(scratch.sp_edge_angles[i] / image_pattern_largest_edge);
                }

                // Min/max edge ration bounds
                scratch.sp_pattern_key_space_min.clear();
                scratch.sp_pattern_key_space_max.clear();
                scratch.sp_image_pattern_key.clear();

                for &ratio in &scratch.sp_image_pattern {
                    let min_val = (ratio - p_max_err).max(0.0) * (p_bins as f64);
                    let max_val = (ratio + p_max_err).min(1.0) * (p_bins as f64);
                    scratch.sp_pattern_key_space_min.push(min_val as usize);
                    scratch.sp_pattern_key_space_max.push(max_val as usize);
                    scratch
                        .sp_image_pattern_key
                        .push((ratio * p_bins as f64) as usize);
                }

                // Generate search space combinations via zero-allocation DFS (replaces Cartesian product)
                scratch.sp_pattern_key_list_fallback.clear();
                scratch.sp_pattern_key_current.clear();
                generate_pattern_keys(
                    &scratch.sp_pattern_key_space_min,
                    &scratch.sp_pattern_key_space_max,
                    &scratch.sp_image_pattern_key,
                    &mut scratch.sp_pattern_key_current,
                    &mut scratch.sp_pattern_key_list_fallback,
                );

                // Explore closest hash codes first
                scratch
                    .sp_pattern_key_list_fallback
                    .sort_unstable_by_key(|k| k.0);
                let mut image_pattern_largest_distance = None;

                for key_idx in 0..scratch.sp_pattern_key_list_fallback.len() {
                    let pattern_key_hash = Self::compute_pattern_key_hash(
                        &scratch.sp_pattern_key_list_fallback[key_idx].1,
                        p_bins,
                    );
                    let hash_index = Self::pattern_key_hash_to_index(
                        pattern_key_hash,
                        (pattern_catalog_flat.len() / p_size) as u64,
                        linear_probe,
                    );

                    Self::get_all_patterns_for_index_inplace(
                        pattern_key_hash,
                        hash_index,
                        image_pattern_largest_edge,
                        options.fov_estimate.map(|x| x.to_radians()),
                        options.fov_max_error.map(|x| x.to_radians()),
                        pattern_catalog_flat,
                        p_size,
                        pattern_key_hashes,
                        pattern_largest_edge,
                        star_table_flat,
                        linear_probe,
                        &mut scratch.sp_hash_match_inds,
                        &mut scratch.sp_cat_edges_list,
                        &mut scratch.sp_cat_vectors_list,
                    );

                    for cat_idx in 0..scratch.sp_cat_edges_list.len() {
                        let catalog_largest_edge =
                            *scratch.sp_cat_edges_list[cat_idx].last().unwrap();
                        let mut valid = true;
                        for i in 0..(scratch.sp_cat_edges_list[cat_idx].len() - 1) {
                            let cat_ratio =
                                scratch.sp_cat_edges_list[cat_idx][i] / catalog_largest_edge;
                            if cat_ratio < scratch.sp_image_pattern[i] - p_max_err
                                || cat_ratio > scratch.sp_image_pattern[i] + p_max_err
                            {
                                valid = false;
                                break;
                            }
                        }
                        if !valid {
                            continue;
                        }

                        // We have a matched pattern! Calculate refined FOV
                        let fov;
                        if options.fov_estimate.is_some() {
                            fov = catalog_largest_edge / image_pattern_largest_edge * fov_initial;
                        } else {
                            if image_pattern_largest_distance.is_none() {
                                for (idx_p, &i) in image_pattern_indices.iter().enumerate() {
                                    scratch.sp_p_cents[idx_p] = image_centroids_undist[i];
                                }
                                let mut max_dist = 0.0;
                                for i in 0..p_size {
                                    for j in (i + 1)..p_size {
                                        let dy =
                                            scratch.sp_p_cents[i][0] - scratch.sp_p_cents[j][0];
                                        let dx =
                                            scratch.sp_p_cents[i][1] - scratch.sp_p_cents[j][1];
                                        let d = (dy * dy + dx * dx).sqrt();
                                        if d > max_dist {
                                            max_dist = d;
                                        }
                                    }
                                }
                                image_pattern_largest_distance = Some(max_dist);
                            }
                            let f = image_pattern_largest_distance.unwrap()
                                / 2.0
                                / (catalog_largest_edge / 2.0).tan();
                            fov = 2.0 * (width / 2.0 / f).atan();
                        }

                        // Re-calculate vectors uniquely sorted by radius to centroid
                        for (idx_p, &i) in image_pattern_indices.iter().enumerate() {
                            scratch.sp_p_cents[idx_p] = image_centroids_undist[i];
                        }

                        compute_vectors_inplace(
                            &scratch.sp_p_cents[..p_size],
                            height,
                            width,
                            fov,
                            &mut scratch.sp_p_vecs,
                            p_size,
                        );
                        sort_vectors_by_radius_inplace(
                            &scratch.sp_p_vecs[..p_size],
                            &mut scratch.sp_image_pattern_vectors_sorted,
                            &mut scratch.sp_radii_scratch,
                            p_size,
                        );

                        if presorted {
                            scratch.sp_catalog_pattern_vectors_sorted[..p_size]
                                .copy_from_slice(&scratch.sp_cat_vectors_list[cat_idx][..p_size]);
                        } else {
                            sort_vectors_by_radius_inplace(
                                &scratch.sp_cat_vectors_list[cat_idx][..p_size],
                                &mut scratch.sp_catalog_pattern_vectors_sorted,
                                &mut scratch.sp_radii_scratch,
                                p_size,
                            );
                        };

                        let (rotation_matrix, det) = find_rotation_matrix_and_det_inplace(
                            &scratch.sp_image_pattern_vectors_sorted[..p_size],
                            &scratch.sp_catalog_pattern_vectors_sorted[..p_size],
                            p_size,
                        );
                        if det < 0.0 {
                            continue;
                        }

                        if let Some(solution) = verify_and_build_solution(
                            scratch,
                            star_kd_tree,
                            star_table_flat,
                            &self.star_catalog_ids,
                            &self.db_props,
                            self.num_patterns,
                            &rotation_matrix,
                            fov,
                            height,
                            width,
                            &options,
                            &image_centroids,
                            &mut image_centroids_undist,
                            num_extracted_stars,
                            match_threshold,
                            t0_solve,
                        ) {
                            return solution;
                        }
                    }
                }
            }
        }

        Solution {
            status: SolveStatus::NoMatch,
            t_solve_ms: t0_solve.elapsed().as_secs_f64() * 1000.0,
            ..Default::default()
        }
    }
}
