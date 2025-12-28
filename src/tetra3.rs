use std::{
    fs::File,
    io::BufReader,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

use anyhow::{Context, Result};
use itertools::Itertools;
use ndarray::{s, Array1, Array2, Axis};
use ndarray_linalg::{Determinant, LeastSquaresSvd, Norm, SVD};
use ndarray_npy::ReadNpyExt;
use scirs2_spatial::{distance::EuclideanDistance, KDTree};
use scirs2_stats::distributions::binomial::Binomial;
use serde::Deserialize;
use zip::ZipArchive;

// Constants
pub const MATCH_FOUND: u8 = 1;
pub const NO_MATCH: u8 = 2;
pub const TIMEOUT: u8 = 3;
pub const CANCELLED: u8 = 4;
pub const TOO_FEW: u8 = 5;

const MAGIC_RAND: u64 = 2654435761;

#[derive(Debug, Clone, Deserialize)]
pub struct DatabaseProperties {
    pub pattern_mode: String,
    pub hash_table_type: String,
    pub pattern_size: usize,
    pub pattern_bins: usize,
    pub pattern_max_error: f64,
    pub max_fov: f64,
    pub min_fov: f64,
    pub star_catalog: String,
    pub epoch_equinox: f64,
    pub epoch_proper_motion: f64,
    pub verification_stars_per_fov: usize,
    pub star_max_magnitude: f64,
    pub presort_patterns: Option<bool>,
    pub num_patterns: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct SolveResult {
    pub ra: Option<f64>,
    pub dec: Option<f64>,
    pub roll: Option<f64>,
    pub fov: Option<f64>,
    pub distortion: Option<f64>,
    pub rmse: Option<f64>,
    pub matches: Option<usize>,
    pub prob: Option<f64>,
    pub t_solve: f64,
    pub status: u8,
    // Additional fields like visual, catalog stars can be added here
}

pub struct Tetra3 {
    star_table: Array2<f64>,
    star_kd_tree: Option<KDTree<f64, EuclideanDistance<f64>>>,
    pattern_catalog: Array2<u16>, // Using u16 for indices
    pattern_largest_edge: Option<Array1<f32>>,
    pattern_key_hashes: Option<Array1<u16>>,
    star_catalog_ids: Option<Array1<u32>>,
    db_props: DatabaseProperties,
    cancelled: AtomicBool,
}

impl Tetra3 {
    pub fn new(load_database: Option<&str>) -> Result<Self> {
        // Default properties if none loaded (though loading is usually
        // required)
        let default_props = DatabaseProperties {
            pattern_mode: "edge_ratio".to_string(),
            hash_table_type: "quadratic_probe".to_string(),
            pattern_size: 4,
            pattern_bins: 50,
            pattern_max_error: 0.005,
            max_fov: 30.0,
            min_fov: 10.0,
            star_catalog: "unknown".to_string(),
            epoch_equinox: 2000.0,
            epoch_proper_motion: 2000.0,
            verification_stars_per_fov: 100,
            star_max_magnitude: 8.0,
            presort_patterns: Some(false),
            num_patterns: None,
        };

        let mut t3 = Tetra3 {
            star_table: Array2::zeros((0, 6)),
            star_kd_tree: None,
            pattern_catalog: Array2::zeros((0, 0)),
            pattern_largest_edge: None,
            pattern_key_hashes: None,
            star_catalog_ids: None,
            db_props: default_props,
            cancelled: AtomicBool::new(false),
        };

        if let Some(path) = load_database {
            t3.load_database(path)?;
        }

        Ok(t3)
    }

    pub fn load_database(&mut self, path_str: &str) -> Result<()> {
        let path = Path::new(path_str);
        let file = File::open(path).context("Failed to open database file")?;
        let reader = BufReader::new(file);
        let mut archive =
            ZipArchive::new(reader).context("Failed to read zip archive")?;

        // Load properties.json
        let props_file = archive
            .by_name("properties.json")
            .context("properties.json not found in archive")?;
        self.db_props = serde_json::from_reader(props_file)?;

        // Helper to read npy from zip
        fn read_npy_f64(
            archive: &mut ZipArchive<BufReader<File>>,
            name: &str,
        ) -> Result<Array2<f64>> {
            let file = archive.by_name(name)?;
            let arr = Array2::<f64>::read_npy(file)?;
            Ok(arr)
        }

        fn read_npy_f32(
            archive: &mut ZipArchive<BufReader<File>>,
            name: &str,
        ) -> Result<Array2<f32>> {
            let file = archive.by_name(name)?;
            let arr = Array2::<f32>::read_npy(file)?;
            Ok(arr)
        }

        fn read_npy_u32_1d(
            archive: &mut ZipArchive<BufReader<File>>,
            name: &str,
        ) -> Result<Array1<u32>> {
            let file = archive.by_name(name)?;
            let arr = Array1::<u32>::read_npy(file)?;
            Ok(arr)
        }

        fn read_npy_u16_2d(
            archive: &mut ZipArchive<BufReader<File>>,
            name: &str,
        ) -> Result<Array2<u16>> {
            let file = archive.by_name(name)?;
            let arr = Array2::<u16>::read_npy(file)?;
            Ok(arr)
        }

        // Load star table
        let tmp_table = read_npy_f32(&mut archive, "star_table.npy")?;
        self.star_table = tmp_table.mapv(|elem| elem as f64);

        // Build KDTree
        let vectors = self.star_table.slice(s![.., 2..5]).to_owned();
        self.star_kd_tree = Some(KDTree::new(&vectors)?);

        // Load Pattern Catalog
        // Note: Python code implies pattern catalog could be compressed or
        // different types. Assuming standard npy here.
        self.pattern_catalog =
            read_npy_u16_2d(&mut archive, "pattern_catalog.npy")?;

        // Optional arrays
        if let Ok(file) = archive.by_name("pattern_largest_edge.npy") {
            self.pattern_largest_edge = Some(Array1::<f32>::read_npy(file)?);
        }

        if let Ok(file) = archive.by_name("pattern_key_hashes.npy") {
            self.pattern_key_hashes = Some(Array1::<u16>::read_npy(file)?);
        }

        if let Ok(file) = archive.by_name("star_catalog_IDs.npy") {
            self.star_catalog_ids = Some(Array1::<u32>::read_npy(file)?);
        }

        Ok(())
    }

    pub fn solve_from_centroids(
        &mut self,
        star_centroids: &Array2<f64>,
        size: (u32, u32),
        fov_estimate: Option<f64>,
        fov_max_error: Option<f64>,
        match_radius: Option<f64>,
        match_threshold: Option<f64>,
        solve_timeout: Option<u64>,
        distortion: Option<f64>,
    ) -> SolveResult {
        let t0 = Instant::now();
        let height = size.0 as f64;
        let width = size.1 as f64;

        // Defaults
        let fov_est_rad =
            fov_estimate.map(|d| d.to_radians()).unwrap_or_else(|| {
                ((self.db_props.max_fov + self.db_props.min_fov) / 2.0)
                    .to_radians()
            });

        let mut fov = fov_est_rad; // Mutable FOV for refinement
        let mut distortion_est = distortion; // Mutable distortion

        let fov_max_err_rad = fov_max_error.map(|d| d.to_radians());
        let match_radius_val = match_radius.unwrap_or(0.01);
        let num_patterns = self
            .db_props
            .num_patterns
            .unwrap_or(self.pattern_catalog.nrows() / 2);
        let match_threshold_val =
            match_threshold.unwrap_or(1e-5) / (num_patterns as f64);

        let p_size = self.db_props.pattern_size;
        let p_bins = self.db_props.pattern_bins;
        let p_max_err = self.db_props.pattern_max_error;
        let linear_probe = self.db_props.hash_table_type == "linear_probe";

        if star_centroids.nrows() < p_size {
            return SolveResult {
                ra: None,
                dec: None,
                roll: None,
                fov: None,
                distortion: None,
                rmse: None,
                matches: None,
                prob: None,
                t_solve: 0.0,
                status: TOO_FEW,
            };
        }

        // Filter centroids (Cluster Buster)

        // 1. Calculate thinning parameters
        // Python: width * .6 * fov / np.sqrt(stars_per_fov) / fov_initial
        let verification_limit = self.db_props.verification_stars_per_fov;
        let separation_pixels =
            width as f64 * 0.6 / (verification_limit as f64).sqrt();

        // 2. Perform Thinning (Cluster Buster)
        let mut keep_for_patterns = vec![false; star_centroids.nrows()];

        // Build KDTree.
        if let Ok(tree) = KDTree::new(star_centroids) {
            for (i, row) in star_centroids.outer_iter().enumerate() {
                // Convert the row (ArrayView) to a slice for the KDTree query
                if let Some(point_slice) = row.as_slice() {
                    if let Ok((neighbor_indices, _)) =
                        tree.query_radius(point_slice, separation_pixels)
                    {
                        // Check if any *already processed* neighbor is kept.
                        let occupied = neighbor_indices
                            .iter()
                            .any(|&idx| keep_for_patterns[idx]);

                        if !occupied {
                            keep_for_patterns[i] = true;
                        }
                    }
                }
            }
        } else {
            // Fallback: keep all if tree construction fails
            keep_for_patterns.fill(true);
        }

        // Collect the indices of stars to use for pattern generation
        let mut pattern_centroids_inds: Vec<usize> = keep_for_patterns
            .iter()
            .enumerate()
            .filter_map(|(i, &keep)| if keep { Some(i) } else { None })
            .collect();

        // 3. Truncate for Verification
        // Create the final image_centroids array used for matching/verification
        let mut image_centroids = star_centroids.clone();
        if image_centroids.nrows() > verification_limit {
            image_centroids = image_centroids
                .slice(s![0..verification_limit, ..])
                .to_owned();
        }
        let num_centroids = image_centroids.nrows();

        // 4. Safety Filter
        // Ensure pattern indices point to valid stars in the (potentially
        // truncated) image_centroids
        pattern_centroids_inds.retain(|&idx| idx < num_centroids);

        // Initial Undistort
        let mut image_centroids_undist = image_centroids.clone();
        if let Some(k) = distortion_est {
            image_centroids_undist =
                undistort_centroids(&image_centroids, size, k);
        }

        // Compute vectors (coarse)
        let image_centroid_vectors =
            compute_vectors(&image_centroids_undist, size, fov);

        // Use the thinned pattern indices for generating combinations
        let combinations =
            pattern_centroids_inds.into_iter().combinations(p_size);

        for pattern_indices in combinations {
            if let Some(ms) = solve_timeout {
                if t0.elapsed().as_millis() as u64 > ms {
                    return SolveResult {
                        ra: None,
                        dec: None,
                        roll: None,
                        fov: None,
                        distortion: None,
                        rmse: None,
                        matches: None,
                        prob: None,
                        t_solve: t0.elapsed().as_secs_f64() * 1000.0,
                        status: TIMEOUT,
                    };
                }
            }
            if self.cancelled.swap(false, Ordering::SeqCst) {
                return SolveResult {
                    ra: None,
                    dec: None,
                    roll: None,
                    fov: None,
                    distortion: None,
                    rmse: None,
                    matches: None,
                    prob: None,
                    t_solve: t0.elapsed().as_secs_f64() * 1000.0,
                    status: CANCELLED,
                };
            }

            // 1. Extract Pattern and Calculate Edges
            let mut pattern_vectors = Array2::<f64>::zeros((p_size, 3));
            for (i, &idx) in pattern_indices.iter().enumerate() {
                pattern_vectors
                    .row_mut(i)
                    .assign(&image_centroid_vectors.row(idx));
            }

            let mut edge_angles = Vec::new();
            for i in 0..p_size {
                for j in (i + 1)..p_size {
                    let d = (&pattern_vectors.row(i) - &pattern_vectors.row(j))
                        .norm_l2();
                    edge_angles.push(angle_from_distance(d));
                }
            }
            edge_angles.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let largest_edge = *edge_angles.last().unwrap();
            let mut edge_ratios = Array1::from_vec(
                edge_angles[0..edge_angles.len() - 1].to_vec(),
            );
            edge_ratios.mapv_inplace(|x| x / largest_edge);

            // 2. Database Lookup
            let edge_ratio_min = &edge_ratios - p_max_err;
            let edge_ratio_max = &edge_ratios + p_max_err;

            let bins_min = edge_ratio_min
                .mapv(|x| (x * p_bins as f64).floor().max(0.0) as u64);
            let bins_max = edge_ratio_max
                .mapv(|x| (x * p_bins as f64).ceil().min(p_bins as f64) as u64);

            let ranges: Vec<Vec<u64>> = bins_min
                .iter()
                .zip(bins_max.iter())
                .map(|(&min, &max)| (min..=max).collect())
                .collect();

            let key_combinations = ranges.into_iter().multi_cartesian_product();
            let image_key = edge_ratios.mapv(|x| (x * p_bins as f64) as i64);

            let mut sorted_keys: Vec<(u64, Vec<u64>)> = key_combinations
                .map(|key| {
                    let dist: i64 = key
                        .iter()
                        .zip(image_key.iter())
                        .map(|(&k, &ik)| (k as i64 - ik).pow(2))
                        .sum();
                    (dist as u64, key)
                })
                .collect();
            sorted_keys.sort_by_key(|k| k.0);

            for (_, pattern_key) in sorted_keys {
                let hash_val =
                    compute_pattern_key_hash(&pattern_key, p_bins as u64);
                let hash_idx = pattern_key_hash_to_index(
                    hash_val,
                    self.pattern_catalog.nrows() as u64,
                    linear_probe,
                );

                let matches = self.get_all_patterns_for_index(
                    hash_val,
                    hash_idx,
                    largest_edge,
                    fov_est_rad,
                    fov_max_err_rad,
                    linear_probe,
                );

                if matches.is_none() {
                    continue;
                }
                let (cat_edges, cat_vectors_list) = matches.unwrap();

                for (idx, cat_edge_row) in cat_edges.outer_iter().enumerate() {
                    let cat_largest = cat_edge_row[cat_edge_row.len() - 1];
                    let cat_ratios =
                        cat_edge_row.slice(s![..-1]).mapv(|x| x / cat_largest);

                    let valid = cat_ratios
                        .iter()
                        .zip(edge_ratio_min.iter())
                        .all(|(c, m)| c > m)
                        && cat_ratios
                            .iter()
                            .zip(edge_ratio_max.iter())
                            .all(|(c, m)| c < m);

                    if !valid {
                        continue;
                    }

                    // 3. Pattern Match Verification
                    if fov_estimate.is_some() {
                        fov = cat_largest / largest_edge * fov_est_rad;
                    } else {
                        let pattern_dist_max = pdist_max(
                            &image_centroids_undist,
                            &pattern_indices,
                        );
                        let f =
                            pattern_dist_max / 2.0 / (cat_largest / 2.0).tan();
                        fov = 2.0 * (width / 2.0 / f).atan();
                    }

                    // Sort pattern stars
                    let img_vec_refined = compute_vectors_indexed(
                        &image_centroids_undist,
                        &pattern_indices,
                        size,
                        fov,
                    );
                    let img_vec_sorted =
                        sort_vectors_by_centroid_dist(img_vec_refined);
                    let mut cat_vec_sorted = cat_vectors_list[idx].clone();
                    if self.db_props.presort_patterns != Some(true) {
                        cat_vec_sorted =
                            sort_vectors_by_centroid_dist(cat_vec_sorted);
                    }

                    let mut rot_mat =
                        find_rotation_matrix(&img_vec_sorted, &cat_vec_sorted);
                    match rot_mat.det() {
                        Ok(n) => {
                            if n < 0.0 {
                                continue;
                            }
                        }
                        Err(_) => {
                            continue;
                        }
                    }

                    // Find all catalog stars in FOV
                    let center_vec = rot_mat.row(0).to_owned();
                    let diag_fov =
                        fov * (width.powi(2) + height.powi(2)).sqrt() / width;
                    let nearby_indices = self
                        .get_nearby_catalog_stars(&center_vec, diag_fov / 2.0);

                    let mut nearby_vectors =
                        Array2::<f64>::zeros((nearby_indices.len(), 3));
                    for (i, &db_idx) in nearby_indices.iter().enumerate() {
                        nearby_vectors.row_mut(i).assign(
                            &self.star_table.row(db_idx).slice(s![2..5]),
                        );
                    }

                    let nearby_derot = nearby_vectors.dot(&rot_mat);
                    let (nearby_centroids, valid_mask) =
                        compute_centroids(&nearby_derot, size, fov);

                    let valid_indices: Vec<usize> = valid_mask
                        .iter()
                        .enumerate()
                        .filter(|&(_, &v)| v)
                        .map(|(i, _)| i)
                        .collect();

                    let limit = (2 * num_centroids).min(valid_indices.len());
                    let check_indices: Vec<usize> =
                        valid_indices.iter().take(limit).cloned().collect();
                    let nearby_centroids_check =
                        nearby_centroids.select(Axis(0), &check_indices);

                    let matches = find_centroid_matches(
                        &image_centroids_undist,
                        &nearby_centroids_check,
                        width * match_radius_val,
                    );

                    let k_matches = matches.len();

                    // Probability Check
                    let prob_single_mismatch = (nearby_indices.len() as f64)
                        * match_radius_val.powi(2);
                    let prob_success = 1.0 - prob_single_mismatch;
                    let matches_adjusted =
                        if k_matches > 2 { k_matches - 2 } else { 0 };
                    let failures = if num_centroids > matches_adjusted {
                        num_centroids - matches_adjusted
                    } else {
                        0
                    };
                    let binom =
                        Binomial::new(num_centroids as usize, prob_success)
                            .expect("Failed to create binomial");
                    let prob_mismatch_val = binom.cdf(failures as f64);

                    if prob_mismatch_val >= match_threshold_val {
                        continue;
                    }

                    // --- MATCH ACCEPTED: REFINEMENT AND OPTIMIZATION ---

                    let match_img_idx: Vec<usize> =
                        matches.iter().map(|m| m.0).collect();
                    let match_cat_local_idx: Vec<usize> =
                        matches.iter().map(|m| m.1).collect();
                    let match_cat_idx: Vec<usize> = match_cat_local_idx
                        .iter()
                        .map(|&loc| nearby_indices[check_indices[loc]])
                        .collect();

                    let mut matched_cat_vecs =
                        Array2::<f64>::zeros((match_cat_idx.len(), 3));
                    for (r, &idx) in match_cat_idx.iter().enumerate() {
                        matched_cat_vecs
                            .row_mut(r)
                            .assign(&self.star_table.row(idx).slice(s![2..5]));
                    }

                    // Get matched image centroids (original, distorted if k !=
                    // 0)
                    let matched_img_centroids =
                        image_centroids.select(Axis(0), &match_img_idx);

                    if distortion_est.is_some() {
                        // Re-calculate Rotation Matrix using all matches
                        // (coarse FOV)
                        let matched_img_vecs_coarse = compute_vectors(
                            &image_centroids_undist
                                .select(Axis(0), &match_img_idx),
                            size,
                            fov,
                        );
                        rot_mat = find_rotation_matrix(
                            &matched_img_vecs_coarse,
                            &matched_cat_vecs,
                        );

                        // Optimization Logic: Solve for f and k
                        // 1. Derotate catalog vectors
                        let matched_cat_vecs_derot =
                            matched_cat_vecs.dot(&rot_mat);

                        // 2. Calculate Tangents (ideal pinhole projection
                        //    radii)
                        let tangents = matched_cat_vecs_derot
                            .map_axis(Axis(1), |v| {
                                (v[1].powi(2) + v[2].powi(2)).sqrt() / v[0]
                            });

                        // 3. Calculate measured radii from image center
                        //    (normalized by width)
                        let center_pixel =
                            Array1::from_vec(vec![height / 2.0, width / 2.0]);
                        let radii =
                            matched_img_centroids.map_axis(Axis(1), |pt| {
                                let dy = pt[0] - center_pixel[0];
                                let dx = pt[1] - center_pixel[1];
                                (dy * dy + dx * dx).sqrt() / width * 2.0
                            });

                        // 4. Build Ax = b
                        // b = radii
                        // A = [tangents, radii^3]
                        let mut a_mat =
                            Array2::<f64>::zeros((tangents.len(), 2));
                        a_mat.column_mut(0).assign(&tangents);
                        a_mat.column_mut(1).assign(&radii.mapv(|r| r.powi(3)));

                        // Solve Least Squares
                        // Requires: use ndarray_linalg::LeastSquaresSvd;
                        match a_mat.least_squares(&radii) {
                            Ok(result) => {
                                let sol = result.solution;
                                let f_fit = sol[0];
                                let k_fit = sol[1];

                                // 5. Update FOV and Distortion
                                // f_fit is focal length in units of width/2
                                // correct f_fit = f_real * (1 - k) approx
                                let f_corrected = f_fit / (1.0 - k_fit);
                                fov = 2.0 * (1.0 / f_corrected).atan();
                                distortion_est = Some(k_fit);

                                // 6. Re-undistort entire image centroids with
                                //    new K
                                image_centroids_undist = undistort_centroids(
                                    &image_centroids,
                                    size,
                                    k_fit,
                                );
                            }
                            Err(_) => {
                                // Fallback or log error if SVD fails (rare)
                            }
                        }
                    } else {
                        // Simple FOV correction if no distortion calculation
                        let matched_img_vecs = compute_vectors(
                            &image_centroids_undist
                                .select(Axis(0), &match_img_idx),
                            size,
                            fov,
                        );

                        // Calculate mean ratio of angles
                        let mut angles_cam = Vec::new();
                        let mut angles_cat = Vec::new();

                        // Sample a subset for speed or do full pdist (doing
                        // simplistic loop here)
                        // Using adjacent pairs for linear complexity O(N)
                        // instead of O(N^2) for speed in rust loop
                        for i in 0..matched_img_vecs.nrows() - 1 {
                            let d_cam = (&matched_img_vecs.row(i)
                                - &matched_img_vecs.row(i + 1))
                                .norm_l2();
                            let d_cat = (&matched_cat_vecs.row(i)
                                - &matched_cat_vecs.row(i + 1))
                                .norm_l2();
                            angles_cam.push(angle_from_distance(d_cam));
                            angles_cat.push(angle_from_distance(d_cat));
                        }

                        if !angles_cam.is_empty() {
                            let ratio_sum: f64 = angles_cat
                                .iter()
                                .zip(angles_cam.iter())
                                .map(|(c, m)| c / m)
                                .sum();
                            let ratio_mean =
                                ratio_sum / angles_cam.len() as f64;
                            fov *= ratio_mean;
                        }
                    }

                    // --- FINAL SOLUTION CALCULATION ---

                    // Recompute vectors with final refined FOV and undistorted
                    // centroids
                    let matched_img_vecs_final = compute_vectors(
                        &image_centroids_undist.select(Axis(0), &match_img_idx),
                        size,
                        fov,
                    );

                    // Final Rotation Matrix
                    let final_rot = find_rotation_matrix(
                        &matched_img_vecs_final,
                        &matched_cat_vecs,
                    );

                    // Extract RA/Dec/Roll
                    let ra = final_rot[[0, 1]]
                        .atan2(final_rot[[0, 0]])
                        .to_degrees()
                        .rem_euclid(360.0);
                    let dec = final_rot[[0, 2]]
                        .atan2(
                            (final_rot[[1, 2]].powi(2)
                                + final_rot[[2, 2]].powi(2))
                            .sqrt(),
                        )
                        .to_degrees();
                    let roll = final_rot[[1, 2]]
                        .atan2(final_rot[[2, 2]])
                        .to_degrees()
                        .rem_euclid(360.0);

                    // RMSE Calculation
                    let final_match_vecs_rotated =
                        matched_img_vecs_final.dot(&final_rot.t());
                    let mut sq_err_sum = 0.0;
                    for i in 0..matched_cat_vecs.nrows() {
                        let diff = (&final_match_vecs_rotated.row(i)
                            - &matched_cat_vecs.row(i))
                            .norm_l2();
                        let angle = angle_from_distance(diff);
                        sq_err_sum += angle.powi(2);
                    }
                    let rmse = (sq_err_sum / matched_cat_vecs.nrows() as f64)
                        .sqrt()
                        .to_degrees()
                        * 3600.0;

                    return SolveResult {
                        ra: Some(ra),
                        dec: Some(dec),
                        roll: Some(roll),
                        fov: Some(fov.to_degrees()),
                        distortion: distortion_est,
                        rmse: Some(rmse),
                        matches: Some(k_matches),
                        prob: Some(prob_mismatch_val * num_patterns as f64),
                        t_solve: t0.elapsed().as_secs_f64() * 1000.0,
                        status: MATCH_FOUND,
                    };
                }
            }
        }

        SolveResult {
            ra: None,
            dec: None,
            roll: None,
            fov: None,
            distortion: None,
            rmse: None,
            matches: None,
            prob: None,
            t_solve: t0.elapsed().as_secs_f64() * 1000.0,
            status: NO_MATCH,
        }
    }

    pub fn set_cancelled(&mut self, cancelled: bool) {
        self.cancelled.store(cancelled, Ordering::SeqCst);
    }

    fn get_all_patterns_for_index(
        &self,
        key_hash: u64,
        hash_idx: u64,
        image_edge: f64,
        fov_est: f64,
        fov_err: Option<f64>,
        linear_probe: bool,
    ) -> Option<(Array2<f64>, Vec<Array2<f64>>)> {
        // Find indices in table
        let max_ind = self.pattern_catalog.nrows() as u64;
        let mut found_indices = Vec::new();

        // Probe loop
        for c in 0.. {
            let idx = if linear_probe {
                (hash_idx + c) % max_ind
            } else {
                (hash_idx + c * c) % max_ind
            } as usize;

            // Check if empty (assuming 0 row means empty)
            let row = self.pattern_catalog.row(idx);
            if row.iter().all(|&x| x == 0) {
                break;
            }

            // Check Hash collision if table available
            if let Some(hashes) = &self.pattern_key_hashes {
                if hashes[idx] != (key_hash & 0xFFFF) as u16 {
                    continue;
                }
            }

            // Check FOV if available
            if let Some(edges) = &self.pattern_largest_edge {
                if let Some(err) = fov_err {
                    let db_edge = edges[idx] as f64;
                    let fov2 = db_edge / image_edge * fov_est;
                    if (fov2 - fov_est).abs() > err {
                        continue;
                    }
                }
            }

            found_indices.push(idx);
        }

        if found_indices.is_empty() {
            return None;
        }

        let p_size = self.db_props.pattern_size;
        // Build result arrays
        // Pattern vectors from star table
        let mut cat_vectors_list = Vec::new();
        // Edges matrix
        let num_edges = p_size * (p_size - 1) / 2;
        let mut cat_edges =
            Array2::<f64>::zeros((found_indices.len(), num_edges + 1)); // +1 for largest edge logic in python

        for (i, &pat_idx) in found_indices.iter().enumerate() {
            let star_indices = self.pattern_catalog.row(pat_idx);
            let mut vectors = Array2::<f64>::zeros((p_size, 3));
            for k in 0..p_size {
                let s_idx = star_indices[k] as usize;
                vectors
                    .row_mut(k)
                    .assign(&self.star_table.row(s_idx).slice(s![2..5]));
            }

            // Calculate edges
            let mut edges = Vec::new();
            for a in 0..p_size {
                for b in (a + 1)..p_size {
                    let d = (&vectors.row(a) - &vectors.row(b)).norm_l2();
                    edges.push(angle_from_distance(d));
                }
            }
            edges.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Store in cat_edges (last element is largest)
            for (k, val) in edges.iter().enumerate() {
                cat_edges[[i, k]] = *val;
            }
            cat_vectors_list.push(vectors);
        }

        Some((cat_edges, cat_vectors_list))
    }

    fn get_nearby_catalog_stars(
        &self,
        vector: &Array1<f64>,
        radius: f64,
    ) -> Vec<usize> {
        if let Some(tree) = &self.star_kd_tree {
            let max_dist = distance_from_angle(radius);
            let vec_slice = vector.as_slice().unwrap();
            if let Ok((indices, _distances)) =
                tree.query_radius(vec_slice, max_dist)
            {
                return indices;
            }
        }
        vec![]
    }
}

// --- Helper Functions ---

fn compute_pattern_key_hash(key: &[u64], bin_factor: u64) -> u64 {
    let mut sum = 0;
    for (i, &val) in key.iter().enumerate() {
        sum += val * bin_factor.pow(i as u32);
    }
    sum
}

fn pattern_key_hash_to_index(hash: u64, max_idx: u64, linear: bool) -> u64 {
    if linear {
        hash % max_idx
    } else {
        (hash.wrapping_mul(MAGIC_RAND)) % max_idx
    }
}

fn angle_from_distance(dist: f64) -> f64 {
    2.0 * (0.5 * dist).asin()
}

fn distance_from_angle(angle: f64) -> f64 {
    2.0 * (angle / 2.0).sin()
}

fn compute_vectors(
    centroids: &Array2<f64>,
    size: (u32, u32),
    fov: f64,
) -> Array2<f64> {
    let (h, w) = (size.0 as f64, size.1 as f64);
    let scale = (fov / 2.0).tan() / w * 2.0;
    let img_center = Array1::from_vec(vec![h / 2.0, w / 2.0]);

    let mut vectors = Array2::<f64>::ones((centroids.nrows(), 3));

    for i in 0..centroids.nrows() {
        let cent = centroids.row(i);
        let diff = &img_center - &cent;
        // Pinhole model: (i, j, k) -> (1, dy, dx) roughly?
        // Python: vectors[:, 2:0:-1] = (img_center - centroids) * scale
        // Python vectors is (N, 3). Index 0 is boresight. 1 is x, 2 is y.
        // Rust implementation needs to match exact axis mapping

        vectors[[i, 1]] = diff[1] * scale; // x
        vectors[[i, 2]] = diff[0] * scale; // y

        // Normalize
        let norm = (vectors[[i, 0]].powi(2)
            + vectors[[i, 1]].powi(2)
            + vectors[[i, 2]].powi(2))
        .sqrt();
        vectors.row_mut(i).mapv_inplace(|x| x / norm);
    }
    vectors
}

fn compute_vectors_indexed(
    centroids: &Array2<f64>,
    indices: &[usize],
    size: (u32, u32),
    fov: f64,
) -> Array2<f64> {
    let sub_centroids = centroids.select(Axis(0), indices);
    compute_vectors(&sub_centroids, size, fov)
}

fn compute_centroids(
    vectors: &Array2<f64>,
    size: (u32, u32),
    fov: f64,
) -> (Array2<f64>, Vec<bool>) {
    let (h, w) = (size.0 as f64, size.1 as f64);
    let scale = -w / 2.0 / (fov / 2.0).tan();

    let mut centroids = Array2::<f64>::zeros((vectors.nrows(), 2));
    let mut valid = Vec::new();

    for i in 0..vectors.nrows() {
        let v = vectors.row(i);
        let boresight = v[0];

        // y = scale * z / x (if x is bore)
        let y = scale * v[2] / boresight + h / 2.0;
        let x = scale * v[1] / boresight + w / 2.0;

        centroids[[i, 0]] = y;
        centroids[[i, 1]] = x;

        if y > 0.0 && y < h && x > 0.0 && x < w {
            valid.push(true);
        } else {
            valid.push(false);
        }
    }
    (centroids, valid)
}

fn undistort_centroids(
    centroids: &Array2<f64>,
    size: (u32, u32),
    k: f64,
) -> Array2<f64> {
    let (h, w) = (size.0 as f64, size.1 as f64);
    let kp = k * (2.0 / w).powi(2);
    let mut res = centroids.clone();

    for i in 0..res.nrows() {
        let y = res[[i, 0]] - h / 2.0;
        let x = res[[i, 1]] - w / 2.0;
        let r2 = y * y + x * x;
        let scale = (1.0 - kp * r2) / (1.0 - k);

        res[[i, 0]] = y * scale + h / 2.0;
        res[[i, 1]] = x * scale + w / 2.0;
    }
    res
}

fn find_rotation_matrix(
    img_vecs: &Array2<f64>,
    cat_vecs: &Array2<f64>,
) -> Array2<f64> {
    // H = img.T * cat
    let h_mat = img_vecs.t().dot(cat_vecs);
    // SVD
    let (u, _, vt) = h_mat.svd(true, true).unwrap();
    let u_mat = u.unwrap();
    let vt_mat = vt.unwrap();

    u_mat.dot(&vt_mat)
}

fn find_centroid_matches(
    img: &Array2<f64>,
    cat: &Array2<f64>,
    r: f64,
) -> Vec<(usize, usize)> {
    let mut matches = Vec::new();
    let mut used_cat = Vec::new(); // Naive uniqueness check

    for i in 0..img.nrows() {
        let p1 = img.row(i);
        for j in 0..cat.nrows() {
            if used_cat.contains(&j) {
                continue;
            }
            let p2 = cat.row(j);
            let dist =
                ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2)).sqrt();
            if dist < r {
                matches.push((i, j));
                used_cat.push(j);
                break; // 1-to-1 assumption per image star
            }
        }
    }
    matches
}

fn sort_vectors_by_centroid_dist(vectors: Array2<f64>) -> Array2<f64> {
    let mut v_vec: Vec<Array1<f64>> =
        vectors.outer_iter().map(|v| v.to_owned()).collect();
    // Centroid of vectors
    let mut center = Array1::<f64>::zeros(3);
    for v in &v_vec {
        center = &center + v;
    }
    center.mapv_inplace(|x| x / v_vec.len() as f64);

    // Sort by dist to center
    v_vec.sort_by(|a, b| {
        let da = (a - &center).norm_l2();
        let db = (b - &center).norm_l2();
        da.partial_cmp(&db).unwrap()
    });

    let mut res = Array2::zeros((vectors.nrows(), 3));
    for (i, v) in v_vec.iter().enumerate() {
        res.row_mut(i).assign(v);
    }
    res
}

fn pdist_max(centroids: &Array2<f64>, indices: &[usize]) -> f64 {
    let mut max_d = 0.0;
    for i in 0..indices.len() {
        for j in (i + 1)..indices.len() {
            let r1 = centroids.row(indices[i]);
            let r2 = centroids.row(indices[j]);
            let d = ((r1[0] - r2[0]).powi(2) + (r1[1] - r2[1]).powi(2)).sqrt();
            if d > max_d {
                max_d = d;
            }
        }
    }
    max_d
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::tetra3::Tetra3;

    #[tokio::test]
    async fn test_database_load() {
        let db_path = Path::new("test/default_database.npz");
        if !db_path.exists() {
            eprintln!("Skipping test: default_database.npz not found.");
            return;
        }
        let _solver = Tetra3::new(db_path.to_str())
            .expect("Failed to load Tetra3 database");
    }
}
