// Copyright (c) 2026 Omair Kamil oakamil@gmail.com
// See LICENSE file in root directory for license terms.

use itertools::Itertools;
use kiddo::{KdTree, SquaredEuclidean};
use nalgebra::{DMatrix, DVector, Matrix3, SVD};
use ndarray::{Array1, Array2, s};
use npyz::NpyFile;
use statrs::distribution::{Binomial, DiscreteCDF};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;
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

// --- High-Performance Math & Projection Helpers ---

fn angle_from_distance(dist: f64) -> f64 {
    2.0 * (0.5 * dist).asin()
}

fn distance_from_angle(angle: f64) -> f64 {
    2.0 * (angle / 2.0).sin()
}

// Computes the exact pairwise distances of an Nx3 array of vectors
fn pdist(vectors: &Array2<f64>) -> Vec<f64> {
    let n = vectors.nrows();
    let mut dists = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let mut sum_sq = 0.0;
            for k in 0..3 {
                let d = vectors[[i, k]] - vectors[[j, k]];
                sum_sq += d * d;
            }
            dists.push(sum_sq.sqrt());
        }
    }
    dists
}

fn compute_vectors(centroids: &Array2<f64>, height: f64, width: f64, fov: f64) -> Array2<f64> {
    let scale_factor = (fov / 2.0).tan() / width * 2.0;
    let img_center_y = height / 2.0;
    let img_center_x = width / 2.0;

    let mut vectors = Array2::<f64>::zeros((centroids.nrows(), 3));
    for (i, row) in centroids.outer_iter().enumerate() {
        let v0 = 1.0;
        let v1 = (img_center_x - row[1]) * scale_factor;
        let v2 = (img_center_y - row[0]) * scale_factor;

        let norm = (v0 * v0 + v1 * v1 + v2 * v2).sqrt();
        vectors[[i, 0]] = v0 / norm;
        vectors[[i, 1]] = v1 / norm;
        vectors[[i, 2]] = v2 / norm;
    }
    vectors
}

fn compute_centroids(
    vectors: &Array2<f64>,
    height: f64,
    width: f64,
    fov: f64,
) -> (Array2<f64>, Vec<usize>) {
    let scale_factor = -width / 2.0 / (fov / 2.0).tan();
    let img_center_y = height / 2.0;
    let img_center_x = width / 2.0;

    let mut centroids = Array2::<f64>::zeros((vectors.nrows(), 2));
    let mut keep = Vec::new();

    for (i, row) in vectors.outer_iter().enumerate() {
        let cy = scale_factor * (row[2] / row[0]) + img_center_y;
        let cx = scale_factor * (row[1] / row[0]) + img_center_x;
        centroids[[i, 0]] = cy;
        centroids[[i, 1]] = cx;

        if cy > 0.0 && cx > 0.0 && cy < height && cx < width {
            keep.push(i);
        }
    }
    (centroids, keep)
}

fn undistort_centroids(centroids: &Array2<f64>, height: f64, width: f64, k: f64) -> Array2<f64> {
    let kp = k * (2.0 / width).powi(2);
    let mut undistorted = centroids.clone();

    for mut row in undistorted.rows_mut() {
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
    centroids: &Array2<f64>,
    height: f64,
    width: f64,
    k: f64,
    tol: f64,
    maxiter: usize,
) -> Array2<f64> {
    let kp = k * (2.0 / width).powi(2);
    let mut distorted = centroids.clone();

    for mut row in distorted.rows_mut() {
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

fn sort_vectors_by_radius(vectors: &Array2<f64>) -> Array2<f64> {
    let n = vectors.nrows();
    let mut centroid = Array1::zeros(3);
    for i in 0..n {
        centroid += &vectors.row(i);
    }
    centroid /= n as f64;

    let mut radii: Vec<(f64, usize)> = (0..n)
        .map(|i| {
            let d = &vectors.row(i) - &centroid;
            let dist = (d[0].powi(2) + d[1].powi(2) + d[2].powi(2)).sqrt();
            (dist, i)
        })
        .collect();

    radii.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut sorted = Array2::zeros((n, 3));
    for (new_idx, &(_, old_idx)) in radii.iter().enumerate() {
        sorted.row_mut(new_idx).assign(&vectors.row(old_idx));
    }
    sorted
}

// Fully unrolled, pure-Rust rotation helper.
fn rotate_vectors(rot: &Array2<f64>, vecs: &Array2<f64>, transpose_rot: bool) -> Array2<f64> {
    let n = vecs.nrows();
    let mut out = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        if transpose_rot {
            out[[i, 0]] = rot[[0, 0]] * vecs[[i, 0]]
                + rot[[1, 0]] * vecs[[i, 1]]
                + rot[[2, 0]] * vecs[[i, 2]];
            out[[i, 1]] = rot[[0, 1]] * vecs[[i, 0]]
                + rot[[1, 1]] * vecs[[i, 1]]
                + rot[[2, 1]] * vecs[[i, 2]];
            out[[i, 2]] = rot[[0, 2]] * vecs[[i, 0]]
                + rot[[1, 2]] * vecs[[i, 1]]
                + rot[[2, 2]] * vecs[[i, 2]];
        } else {
            out[[i, 0]] = rot[[0, 0]] * vecs[[i, 0]]
                + rot[[0, 1]] * vecs[[i, 1]]
                + rot[[0, 2]] * vecs[[i, 2]];
            out[[i, 1]] = rot[[1, 0]] * vecs[[i, 0]]
                + rot[[1, 1]] * vecs[[i, 1]]
                + rot[[1, 2]] * vecs[[i, 2]];
            out[[i, 2]] = rot[[2, 0]] * vecs[[i, 0]]
                + rot[[2, 1]] * vecs[[i, 1]]
                + rot[[2, 2]] * vecs[[i, 2]];
        }
    }
    out
}

// Pure-Rust stack-allocated SVD (via nalgebra).
fn find_rotation_matrix_and_det(
    image_vectors: &Array2<f64>,
    catalog_vectors: &Array2<f64>,
) -> (Array2<f64>, f64) {
    let n = image_vectors.nrows();
    let mut h = Matrix3::<f64>::zeros();

    // H = image_vectors.T * catalog_vectors
    for i in 0..n {
        for r in 0..3 {
            for c in 0..3 {
                h[(r, c)] += image_vectors[[i, r]] * catalog_vectors[[i, c]];
            }
        }
    }

    let svd = SVD::new(h, true, true);
    let u = svd.u.unwrap();
    let vt = svd.v_t.unwrap();
    let rot = u * vt;
    let det = rot.determinant();

    let mut res = Array2::zeros((3, 3));
    for r in 0..3 {
        for c in 0..3 {
            res[[r, c]] = rot[(r, c)];
        }
    }
    (res, det)
}

fn find_centroid_matches(
    image_centroids: &Array2<f64>,
    catalog_centroids: &Array2<f64>,
    r: f64,
) -> Vec<(usize, usize)> {
    let mut matches = Vec::new();
    for i in 0..image_centroids.nrows() {
        for j in 0..catalog_centroids.nrows() {
            let dy = image_centroids[[i, 0]] - catalog_centroids[[j, 0]];
            let dx = image_centroids[[i, 1]] - catalog_centroids[[j, 1]];
            let d = (dy * dy + dx * dx).sqrt();
            if d < r {
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
    let matches2: Vec<(usize, usize)> = indices0.into_iter().map(|idx| matches1[idx]).collect();

    matches2
}

// --- Ported Utility Functions ---

fn separation_for_density(fov: f64, stars_per_fov: f64) -> f64 {
    0.6 * fov / stars_per_fov.sqrt()
}

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

// --- Main Engine ---

pub struct Tetra3 {
    pub star_table: Array2<f64>,
    pub star_kd_tree: KdTree<f64, 3>,
    pub pattern_catalog: Array2<usize>,
    pub pattern_largest_edge: Option<Array1<f32>>,
    pub pattern_key_hashes: Option<Array1<u16>>,
    pub star_catalog_ids: Option<Array1<u32>>,
    pub db_props: HashMap<String, f64>,
    pub num_patterns: usize,
    pub linear_probe: bool,
    cancelled: bool,
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
                let data: Vec<f32> = npy.into_vec().ok()?;
                Some(Array1::from_vec(data))
            })
        };

        let read_1d_u16 = |arc: &mut ZipArchive<File>, name: &str| -> Option<Array1<u16>> {
            arc.by_name(name).ok().and_then(|mut zf| {
                let mut buf = Vec::new();
                zf.read_to_end(&mut buf).ok()?;
                let mut cursor = Cursor::new(&buf);
                let npy = NpyFile::new(&mut cursor).ok()?;
                let data: Vec<u16> = npy.into_vec().ok()?;
                Some(Array1::from_vec(data))
            })
        };

        let read_1d_u32 = |arc: &mut ZipArchive<File>, name: &str| -> Option<Array1<u32>> {
            arc.by_name(name).ok().and_then(|mut zf| {
                let mut buf = Vec::new();
                zf.read_to_end(&mut buf).ok()?;
                let mut cursor = Cursor::new(&buf);
                let npy = NpyFile::new(&mut cursor).ok()?;
                let data: Vec<u32> = npy.into_vec().ok()?;
                Some(Array1::from_vec(data))
            })
        };

        let pattern_catalog = read_pattern_catalog(&mut archive, "pattern_catalog.npy")?;
        let star_table = read_star_table(&mut archive, "star_table.npy")?;
        let pattern_largest_edge = read_1d_f32(&mut archive, "pattern_largest_edge.npy");
        let pattern_key_hashes = read_1d_u16(&mut archive, "pattern_key_hashes.npy");
        let star_catalog_ids = read_1d_u32(&mut archive, "star_catalog_IDs.npy");

        let mut star_kd_tree = KdTree::new();
        for (i, row) in star_table.outer_iter().enumerate() {
            let point = [row[2], row[3], row[4]];
            star_kd_tree.add(&point, i as u64);
        }

        let mut num_patterns = pattern_catalog.nrows() / 2;

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

                        let extracted_num_patterns =
                            u32::from_le_bytes([data[824], data[825], data[826], data[827]]);
                        num_patterns = extracted_num_patterns as usize;
                    }
                }
            }
        }

        Ok(Tetra3 {
            star_table,
            star_kd_tree,
            pattern_catalog,
            pattern_largest_edge,
            pattern_key_hashes,
            star_catalog_ids,
            db_props,
            num_patterns,
            linear_probe,
            cancelled: false,
        })
    }

    pub fn cancel_solve(&mut self) {
        self.cancelled = true;
    }

    fn compute_pattern_key_hash(&self, pattern_key: &[usize], bin_factor: usize) -> u64 {
        let mut hash: u64 = 0;
        let mut multiplier: u64 = 1;
        for &k in pattern_key {
            hash += (k as u64) * multiplier;
            multiplier *= bin_factor as u64;
        }
        hash
    }

    fn pattern_key_hash_to_index(&self, hash: u64, max_index: u64) -> u64 {
        if self.linear_probe {
            hash % max_index
        } else {
            hash.wrapping_mul(MAGIC_RAND) % max_index
        }
    }

    fn get_table_indices_from_hash(&self, hash_index: u64) -> Vec<usize> {
        let max_ind = self.pattern_catalog.nrows() as u64;
        let mut found = Vec::new();
        for c in 0.. {
            let i = if self.linear_probe {
                (hash_index + c) % max_ind
            } else {
                (hash_index + c * c) % max_ind
            };

            let row = self.pattern_catalog.row(i as usize);
            if row.iter().all(|&x| x == 0) {
                break;
            }
            found.push(i as usize);
        }
        found
    }

    fn get_all_patterns_for_index(
        &self,
        pattern_key_hash: u64,
        hash_index: u64,
        image_pattern_largest_edge: f64,
        fov_estimate: Option<f64>,
        fov_max_error: Option<f64>,
    ) -> (Vec<Vec<f64>>, Vec<Array2<f64>>) {
        let mut hash_match_inds = self.get_table_indices_from_hash(hash_index);
        if hash_match_inds.is_empty() {
            return (vec![], vec![]);
        }

        if let Some(hashes) = &self.pattern_key_hashes {
            let key_hash16 = (pattern_key_hash & 0xffff) as u16;
            hash_match_inds.retain(|&idx| hashes[idx] == key_hash16);
        }

        if let (Some(largest_edges), Some(f_est), Some(f_err)) =
            (&self.pattern_largest_edge, fov_estimate, fov_max_error)
        {
            hash_match_inds.retain(|&idx| {
                let cat_largest_edge = largest_edges[idx] as f64;
                let fov2 = cat_largest_edge / image_pattern_largest_edge * f_est / 1000.0;
                (fov2 - f_est).abs() < f_err
            });
        }

        let p_size = self.pattern_catalog.ncols();
        let mut catalog_pattern_edges = Vec::with_capacity(hash_match_inds.len());
        let mut catalog_pattern_vectors = Vec::with_capacity(hash_match_inds.len());

        for &idx in &hash_match_inds {
            let row = self.pattern_catalog.row(idx);
            let mut vecs = Array2::zeros((p_size, 3));
            for i in 0..p_size {
                let star_id = row[i];
                vecs.row_mut(i)
                    .assign(&self.star_table.slice(s![star_id, 2..5]));
            }

            let dists = pdist(&vecs);
            let mut angles: Vec<f64> = dists.into_iter().map(angle_from_distance).collect();
            angles.sort_by(|a, b| a.partial_cmp(b).unwrap());

            catalog_pattern_edges.push(angles);
            catalog_pattern_vectors.push(vecs);
        }

        (catalog_pattern_edges, catalog_pattern_vectors)
    }

    pub fn solve_from_centroids(
        &mut self,
        star_centroids: &Array2<f64>,
        size: (f64, f64),
        options: SolveOptions,
    ) -> Solution {
        let t0_solve = Instant::now();
        let (height, width) = size;

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

        let mut num_centroids = star_centroids.nrows();
        if num_centroids < p_size {
            return Solution {
                status: SolveStatus::TooFew,
                ..Default::default()
            };
        }

        // Thinning Strategy
        let pattern_stars_separation_pixels =
            width * separation_for_density(fov_initial, verification_stars as f64) / fov_initial;
        let mut keep_for_patterns = vec![false; num_centroids];

        for i in 0..num_centroids {
            let mut occupied = false;
            let c_i = star_centroids.row(i);
            for j in 0..i {
                if keep_for_patterns[j] {
                    let c_j = star_centroids.row(j);
                    let dist = ((c_i[0] - c_j[0]).powi(2) + (c_i[1] - c_j[1]).powi(2)).sqrt();
                    if dist < pattern_stars_separation_pixels {
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

        if num_centroids > verification_stars {
            num_centroids = verification_stars;
            pattern_centroids_inds.retain(|&i| i < num_centroids);
        }
        let num_extracted_stars = num_centroids;

        // Maintain the original full set of image_centroids for the final matrix building
        let image_centroids = star_centroids.slice(s![..num_centroids, ..]).to_owned();

        let mut image_centroids_undist = match options.distortion {
            Some(k) => undistort_centroids(&image_centroids, height, width, k),
            None => image_centroids.clone(),
        };

        let image_centroids_vectors =
            compute_vectors(&image_centroids_undist, height, width, fov_initial);

        for image_pattern_indices in breadth_first_combinations(&pattern_centroids_inds, p_size) {
            if let Some(timeout) = options.solve_timeout_ms {
                if t0_solve.elapsed().as_secs_f64() * 1000.0 > timeout {
                    return Solution {
                        status: SolveStatus::Timeout,
                        ..Default::default()
                    };
                }
            }
            if self.cancelled {
                self.cancelled = false;
                return Solution {
                    status: SolveStatus::Cancelled,
                    ..Default::default()
                };
            }

            let mut pattern_vecs = Array2::<f64>::zeros((p_size, 3));
            for (idx, &i) in image_pattern_indices.iter().enumerate() {
                pattern_vecs
                    .row_mut(idx)
                    .assign(&image_centroids_vectors.row(i));
            }

            // Calculate edge angles and sort
            let dists = pdist(&pattern_vecs);
            let mut edge_angles: Vec<f64> = dists.into_iter().map(angle_from_distance).collect();
            edge_angles.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let image_pattern_largest_edge = *edge_angles.last().unwrap();
            let mut image_pattern = Vec::with_capacity(edge_angles.len() - 1);
            for i in 0..(edge_angles.len() - 1) {
                image_pattern.push(edge_angles[i] / image_pattern_largest_edge);
            }

            // Min/max edge ration bounds
            let mut pattern_key_space_min = Vec::with_capacity(image_pattern.len());
            let mut pattern_key_space_max = Vec::with_capacity(image_pattern.len());
            let mut image_pattern_key = Vec::with_capacity(image_pattern.len());

            for &ratio in &image_pattern {
                let min_val = (ratio - p_max_err).max(0.0) * (p_bins as f64);
                let max_val = (ratio + p_max_err).min(1.0) * (p_bins as f64);
                pattern_key_space_min.push(min_val as usize);
                pattern_key_space_max.push(max_val as usize);
                image_pattern_key.push((ratio * p_bins as f64) as usize);
            }

            // Generate search space combinations via Cartesian product
            let ranges: Vec<_> = pattern_key_space_min
                .iter()
                .zip(pattern_key_space_max.iter())
                .map(|(&l, &h)| l..=h)
                .collect();

            let mut pattern_key_list: Vec<(usize, Vec<usize>)> = ranges
                .into_iter()
                .multi_cartesian_product()
                .map(|code| {
                    let mut dist = 0;
                    for i in 0..code.len() {
                        let diff = (code[i] as isize - image_pattern_key[i] as isize).abs();
                        dist += diff * diff;
                    }
                    (dist as usize, code)
                })
                .collect();

            // Explore closest hash codes first
            pattern_key_list.sort_by_key(|k| k.0);
            let mut image_pattern_largest_distance = None;

            for (_, pattern_key) in pattern_key_list {
                let pattern_key_hash = self.compute_pattern_key_hash(&pattern_key, p_bins);
                let hash_index = self.pattern_key_hash_to_index(
                    pattern_key_hash,
                    self.pattern_catalog.nrows() as u64,
                );

                let (cat_edges_list, cat_vectors_list) = self.get_all_patterns_for_index(
                    pattern_key_hash,
                    hash_index,
                    image_pattern_largest_edge,
                    options.fov_estimate.map(|x| x.to_radians()),
                    options.fov_max_error.map(|x| x.to_radians()),
                );

                for (idx, cat_edges) in cat_edges_list.iter().enumerate() {
                    let catalog_largest_edge = *cat_edges.last().unwrap();
                    let mut valid = true;
                    for i in 0..(cat_edges.len() - 1) {
                        let cat_ratio = cat_edges[i] / catalog_largest_edge;
                        if cat_ratio < image_pattern[i] - p_max_err
                            || cat_ratio > image_pattern[i] + p_max_err
                        {
                            valid = false;
                            break;
                        }
                    }
                    if !valid {
                        continue;
                    }

                    // We have a matched pattern! Calculate refined FOV
                    let mut fov;
                    if options.fov_estimate.is_some() {
                        fov = catalog_largest_edge / image_pattern_largest_edge * fov_initial;
                    } else {
                        if image_pattern_largest_distance.is_none() {
                            let mut p_cents = Array2::zeros((p_size, 2));
                            for (idx_p, &i) in image_pattern_indices.iter().enumerate() {
                                p_cents
                                    .row_mut(idx_p)
                                    .assign(&image_centroids_undist.row(i));
                            }
                            let mut max_dist = 0.0;
                            for i in 0..p_size {
                                for j in i + 1..p_size {
                                    let d = ((p_cents[[i, 0]] - p_cents[[j, 0]]).powi(2)
                                        + (p_cents[[i, 1]] - p_cents[[j, 1]]).powi(2))
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
                    let mut p_cents = Array2::zeros((p_size, 2));
                    for (idx_p, &i) in image_pattern_indices.iter().enumerate() {
                        p_cents
                            .row_mut(idx_p)
                            .assign(&image_centroids_undist.row(i));
                    }
                    let p_vecs = compute_vectors(&p_cents, height, width, fov);

                    let image_pattern_vectors_sorted = sort_vectors_by_radius(&p_vecs);

                    let catalog_pattern_vectors_sorted = if presorted {
                        cat_vectors_list[idx].clone()
                    } else {
                        sort_vectors_by_radius(&cat_vectors_list[idx])
                    };

                    let (rotation_matrix, det) = find_rotation_matrix_and_det(
                        &image_pattern_vectors_sorted,
                        &catalog_pattern_vectors_sorted,
                    );
                    if det < 0.0 {
                        continue;
                    }

                    // Find all catalog stars inside the FOV diagonal
                    let fov_diagonal_rad = fov * ((width * width + height * height).sqrt() / width);
                    let image_center_vector = [
                        rotation_matrix[[0, 0]],
                        rotation_matrix[[0, 1]],
                        rotation_matrix[[0, 2]],
                    ];

                    // Epsilon to capture borders safely in f64
                    let max_dist_sq = distance_from_angle(fov_diagonal_rad / 2.0).powi(2) + 1e-8;
                    let mut nearby_cat_stars_inds: Vec<usize> = self
                        .star_kd_tree
                        .within::<SquaredEuclidean>(&image_center_vector, max_dist_sq)
                        .into_iter()
                        .map(|n| n.item as usize)
                        .collect();

                    // Re-sort KDTree return list by index to prioritize brighter stars exactly like Python
                    nearby_cat_stars_inds.sort_unstable();

                    let num_nearby_catalog_stars = nearby_cat_stars_inds.len();
                    if num_nearby_catalog_stars == 0 {
                        continue;
                    }

                    let mut nearby_cat_star_vectors =
                        Array2::<f64>::zeros((num_nearby_catalog_stars, 3));
                    for (idx, &star_idx) in nearby_cat_stars_inds.iter().enumerate() {
                        nearby_cat_star_vectors
                            .row_mut(idx)
                            .assign(&self.star_table.slice(s![star_idx, 2..5]));
                    }

                    let nearby_cat_star_vectors_derot =
                        rotate_vectors(&rotation_matrix, &nearby_cat_star_vectors, false);
                    let (nearby_cat_star_centroids_all, kept) =
                        compute_centroids(&nearby_cat_star_vectors_derot, height, width, fov);

                    let crop_len = kept.len().min(2 * num_centroids);

                    let mut valid_cat_centroids = Array2::zeros((crop_len, 2));
                    let mut valid_cat_vectors = Array2::zeros((crop_len, 3));
                    let mut valid_cat_inds = Vec::with_capacity(crop_len);

                    for (idx, &i) in kept.iter().take(crop_len).enumerate() {
                        valid_cat_centroids
                            .row_mut(idx)
                            .assign(&nearby_cat_star_centroids_all.row(i));
                        valid_cat_vectors
                            .row_mut(idx)
                            .assign(&nearby_cat_star_vectors.row(i));
                        valid_cat_inds.push(nearby_cat_stars_inds[i]);
                    }

                    let matched_stars = find_centroid_matches(
                        &image_centroids_undist,
                        &valid_cat_centroids,
                        width * options.match_radius,
                    );

                    let num_star_matches = matched_stars.len();

                    // Probability Calculation
                    let prob_single_star_mismatch =
                        (crop_len as f64) * options.match_radius.powi(2);
                    let p_raw = 1.0 - prob_single_star_mismatch;
                    let k_raw = num_extracted_stars as i64 - (num_star_matches as i64 - 2);

                    // Safe NaN bypass replicating scipy.stats.binom.cdf behavior
                    let prob_mismatch = if p_raw <= 0.0 || p_raw >= 1.0 || k_raw < 0 {
                        0.0
                    } else {
                        match Binomial::new(p_raw, num_extracted_stars as u64) {
                            Ok(b) => b.cdf(k_raw as u64),
                            Err(_) => 0.0,
                        }
                    };

                    if prob_mismatch >= match_threshold {
                        continue;
                    }

                    // We passed all checks. Complete the final exact solution details
                    let mut matched_img_cents = Array2::zeros((num_star_matches, 2));
                    let mut matched_cat_vecs = Array2::zeros((num_star_matches, 3));
                    for (i, &(img_idx, cat_idx)) in matched_stars.iter().enumerate() {
                        matched_img_cents
                            .row_mut(i)
                            .assign(&image_centroids_undist.row(img_idx));
                        matched_cat_vecs
                            .row_mut(i)
                            .assign(&valid_cat_vectors.row(cat_idx));
                    }

                    let matched_img_vecs = compute_vectors(&matched_img_cents, height, width, fov);

                    let (precise_rotation_matrix, _) =
                        find_rotation_matrix_and_det(&matched_img_vecs, &matched_cat_vecs);

                    let mut k_final = options.distortion;
                    if options.distortion.is_some() {
                        // Refine fov & distortion using Least Squares System
                        // A = [tangent, radius^3], b = [radius]
                        // Note: To fully map lstsq in Rust precisely, build A and B for all matched_stars
                        let mut a_na = DMatrix::<f64>::zeros(num_star_matches, 2);
                        let mut b_na = DVector::<f64>::zeros(num_star_matches);

                        let derotated_matched_cat =
                            rotate_vectors(&precise_rotation_matrix, &matched_cat_vecs, false);

                        for (i, &(img_idx, _)) in matched_stars.iter().enumerate() {
                            let r_cent = &image_centroids.row(img_idx);
                            let r_dist = ((r_cent[0] - height / 2.0).powi(2)
                                + (r_cent[1] - width / 2.0).powi(2))
                            .sqrt()
                                / width
                                * 2.0;
                            let cat_derot = &derotated_matched_cat.row(i);
                            let tangent =
                                (cat_derot[1].powi(2) + cat_derot[2].powi(2)).sqrt() / cat_derot[0];

                            a_na[(i, 0)] = tangent;
                            a_na[(i, 1)] = r_dist.powi(3);
                            b_na[i] = r_dist;
                        }

                        // Pure-Rust SVD Pseudo-Inverse for Distortions
                        let svd = SVD::new(a_na, true, true);
                        if let Ok(pseudo_inv) = svd.pseudo_inverse(1e-7) {
                            let sol = pseudo_inv * b_na;
                            let f_val = sol[0] / (1.0 - sol[1]);
                            k_final = Some(sol[1]);
                            fov = 2.0 * (1.0 / f_val).atan();
                            image_centroids_undist =
                                undistort_centroids(&image_centroids, height, width, sol[1]);

                            for (i, &(img_idx, _)) in matched_stars.iter().enumerate() {
                                matched_img_cents
                                    .row_mut(i)
                                    .assign(&image_centroids_undist.row(img_idx));
                            }
                        }
                    }

                    let final_match_vectors =
                        compute_vectors(&matched_img_cents, height, width, fov);

                    let final_derotated =
                        rotate_vectors(&precise_rotation_matrix, &final_match_vectors, true);

                    let mut distances: Vec<f64> = (0..num_star_matches)
                        .map(|i| {
                            let row_f = final_derotated.row(i);
                            let row_c = matched_cat_vecs.row(i);
                            ((row_f[0] - row_c[0]).powi(2)
                                + (row_f[1] - row_c[1]).powi(2)
                                + (row_f[2] - row_c[2]).powi(2))
                            .sqrt()
                        })
                        .collect();
                    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    let p90_idx = (0.9 * (distances.len() - 1) as f64) as usize;
                    let p90_err_angle =
                        angle_from_distance(distances[p90_idx]).to_degrees() * 3600.0;
                    let max_err_angle =
                        angle_from_distance(*distances.last().unwrap()).to_degrees() * 3600.0;

                    let mut rms_sum = 0.0;
                    for &d in &distances {
                        let a = angle_from_distance(d);
                        rms_sum += a * a;
                    }
                    let rms_err_angle =
                        (rms_sum / distances.len() as f64).sqrt().to_degrees() * 3600.0;

                    let ra = precise_rotation_matrix[[0, 1]]
                        .atan2(precise_rotation_matrix[[0, 0]])
                        .to_degrees()
                        .rem_euclid(360.0);
                    let dec = precise_rotation_matrix[[0, 2]]
                        .atan2(
                            (precise_rotation_matrix[[1, 2]].powi(2)
                                + precise_rotation_matrix[[2, 2]].powi(2))
                            .sqrt(),
                        )
                        .to_degrees();
                    let roll = precise_rotation_matrix[[1, 2]]
                        .atan2(precise_rotation_matrix[[2, 2]])
                        .to_degrees()
                        .rem_euclid(360.0);

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
                        prob: Some(prob_mismatch * (self.num_patterns as f64)),
                        epoch_equinox: self.db_props.get("epoch_equinox").cloned(),
                        epoch_proper_motion: self.db_props.get("epoch_proper_motion").cloned(),
                        status: SolveStatus::MatchFound,
                        t_solve_ms: t0_solve.elapsed().as_secs_f64() * 1000.0,
                        ..Default::default()
                    };

                    if options.return_rotation_matrix {
                        solution.rotation_matrix = Some(precise_rotation_matrix.clone());
                    }

                    if let Some(mut target_px) = options.target_pixel.clone() {
                        if let Some(k) = k_final {
                            target_px = undistort_centroids(&target_px, height, width, k);
                        }
                        let target_vector = compute_vectors(&target_px, height, width, fov);

                        let rotated_target_vector =
                            rotate_vectors(&precise_rotation_matrix, &target_vector, true);

                        let mut target_ra = Vec::new();
                        let mut target_dec = Vec::new();
                        for i in 0..rotated_target_vector.nrows() {
                            let ra_ang = rotated_target_vector[[i, 1]]
                                .atan2(rotated_target_vector[[i, 0]])
                                .to_degrees()
                                .rem_euclid(360.0);
                            let dec_ang = 90.0 - rotated_target_vector[[i, 2]].acos().to_degrees();
                            target_ra.push(ra_ang);
                            target_dec.push(dec_ang);
                        }
                        solution.target_ra = Some(target_ra);
                        solution.target_dec = Some(target_dec);
                    }

                    if let Some(target_sky) = &options.target_sky_coord {
                        let mut target_sky_vecs = Array2::zeros((target_sky.nrows(), 3));
                        for (i, row) in target_sky.outer_iter().enumerate() {
                            let ra_rad = row[0].to_radians();
                            let dec_rad = row[1].to_radians();
                            target_sky_vecs[[i, 0]] = ra_rad.cos() * dec_rad.cos();
                            target_sky_vecs[[i, 1]] = ra_rad.sin() * dec_rad.cos();
                            target_sky_vecs[[i, 2]] = dec_rad.sin();
                        }

                        let target_sky_vecs_derot =
                            rotate_vectors(&precise_rotation_matrix, &target_sky_vecs, false);
                        let (mut target_centroids, kept_sky) =
                            compute_centroids(&target_sky_vecs_derot, height, width, fov);

                        if let Some(k) = k_final {
                            for &idx in &kept_sky {
                                let distorted = distort_centroids(
                                    &target_centroids.slice(s![idx..idx + 1, ..]).to_owned(),
                                    height,
                                    width,
                                    k,
                                    1e-6,
                                    30,
                                );
                                target_centroids.row_mut(idx).assign(&distorted.row(0));
                            }
                        }

                        let mut target_y = vec![None; target_sky.nrows()];
                        let mut target_x = vec![None; target_sky.nrows()];
                        for &idx in &kept_sky {
                            target_y[idx] = Some(target_centroids[[idx, 0]]);
                            target_x[idx] = Some(target_centroids[[idx, 1]]);
                        }
                        solution.target_y = Some(target_y);
                        solution.target_x = Some(target_x);
                    }

                    if options.return_matches {
                        let mut m_cents = Vec::new();
                        let mut m_stars = Vec::new();
                        let mut m_ids = Vec::new();

                        for &(img_idx, cat_idx) in &matched_stars {
                            let img_c = image_centroids_undist.row(img_idx);
                            m_cents.push([img_c[0], img_c[1]]);

                            let star_idx = valid_cat_inds[cat_idx];
                            let cat_row = self.star_table.row(star_idx);
                            m_stars.push([
                                cat_row[0].to_degrees(),
                                cat_row[1].to_degrees(),
                                cat_row[5],
                            ]);

                            if let Some(ids) = &self.star_catalog_ids {
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
                        for (idx, centroid) in valid_cat_centroids.outer_iter().enumerate() {
                            let star_idx = valid_cat_inds[idx];
                            let ra_deg = self.star_table[[star_idx, 0]].to_degrees();
                            let dec_deg = self.star_table[[star_idx, 1]].to_degrees();
                            let mag = self.star_table[[star_idx, 5]];
                            cat_stars.push((ra_deg, dec_deg, mag, centroid[0], centroid[1]));
                        }
                        solution.catalog_stars = Some(cat_stars);
                    }

                    return solution;
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
