// Copyright (c) 2026 Omair Kamil oakamil@gmail.com
// See LICENSE file in root directory for license terms.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering as AtomicOrdering},
    },
    time::Duration,
};

use async_trait::async_trait;
use canonical_error::{
    CanonicalError, deadline_exceeded_error, invalid_argument_error, not_found_error,
};
use cedar_elements::{
    cedar::{ImageCoord, PlateSolution},
    cedar_common::CelestialCoord,
    imu_trait::EquatorialCoordinates,
    solver_trait::{SolveExtension, SolveParams, SolverTrait},
};
use ndarray::Array2;

use tetra3::{SolveOptions, SolveStatus, Tetra3};

pub struct Tetra3Solver {
    inner: tokio::sync::Mutex<Tetra3>,
    // Shared cancellation flag between the trait and the solver loop
    cancelled: Arc<AtomicBool>,
}

impl Tetra3Solver {
    pub fn new(tetra3: Tetra3) -> Self {
        Tetra3Solver {
            inner: tokio::sync::Mutex::new(tetra3),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }
}

#[async_trait]
impl SolverTrait for Tetra3Solver {
    fn cancel(&self) {
        self.cancelled.store(true, AtomicOrdering::Relaxed);
    }

    fn default_timeout(&self) -> Duration {
        Duration::from_secs(5)
    }

    async fn solve_from_centroids(
        &self,
        star_centroids: &[ImageCoord],
        width: usize,
        height: usize,
        extension: &SolveExtension,
        params: &SolveParams,
        _imu_estimate: Option<EquatorialCoordinates>,
    ) -> Result<PlateSolution, CanonicalError> {
        let mut tetra3 = self.inner.lock().await;

        // Convert slice of struct coordinates into the required ndarray Matrix
        // Tetra3 primarily maps image vectors as N x 2 (y, x).
        let mut flat_buffer = Vec::with_capacity(star_centroids.len() * 2);
        for c in star_centroids {
            flat_buffer.push(c.y as f64);
            flat_buffer.push(c.x as f64);
        }
        let centroids_array = Array2::from_shape_vec((star_centroids.len(), 2), flat_buffer)
            .unwrap_or_else(|_| Array2::zeros((0, 2)));

        // Map target pixel array from Vec<ImageCoord> to Option<Array2>
        let target_pixel = extension.target_pixel.as_ref().map(|tp| {
            let mut flat = Vec::with_capacity(tp.len() * 2);
            for c in tp {
                flat.push(c.y as f64);
                flat.push(c.x as f64);
            }
            Array2::from_shape_vec((tp.len(), 2), flat).unwrap_or_else(|_| Array2::zeros((0, 2)))
        });

        // Map target sky coordinate array from Vec<CelestialCoord> to Option<Array2>
        let target_sky_coord = extension.target_sky_coord.as_ref().map(|tsc| {
            let mut flat = Vec::with_capacity(tsc.len() * 2);
            for c in tsc {
                flat.push(c.ra);
                flat.push(c.dec);
            }
            Array2::from_shape_vec((tsc.len(), 2), flat).unwrap_or_else(|_| Array2::zeros((0, 2)))
        });

        // Construct standard parameters dynamically (using precise available fields & defaults)
        let options = SolveOptions {
            fov_estimate: params.fov_estimate.map(|(fov, _)| fov),
            fov_max_error: params.fov_estimate.map(|(_, err)| err).or(Some(0.1)),
            match_radius: params.match_radius.unwrap_or(0.01),
            match_threshold: params.match_threshold.unwrap_or(1e-4),
            // Bypassing Duration vs f64 ambiguity on traits using a static safe max timeout
            solve_timeout_ms: Some(5000.0),
            distortion: params.distortion,
            match_max_error: params.match_max_error.unwrap_or(0.005),
            return_matches: extension.return_matches,
            return_catalog: extension.return_catalog,
            return_rotation_matrix: extension.return_rotation_matrix,
            target_pixel,
            target_sky_coord,
        };

        // Pass the properly mapped array into the solver
        let result =
            tetra3.solve_from_centroids(&centroids_array, (height as f64, width as f64), options);

        match result.status {
            SolveStatus::MatchFound => {
                // Convert the raw f64 milliseconds into a standard Duration,
                // then convert that into the required Protobuf Duration struct.
                let solve_duration = Duration::from_secs_f64(result.t_solve_ms / 1000.0);

                // Populate the correct `PlateSolution` fields cleanly
                Ok(PlateSolution {
                    image_sky_coord: Some(CelestialCoord {
                        ra: result.ra.unwrap_or(0.0),
                        dec: result.dec.unwrap_or(0.0),
                    }),
                    roll: result.roll.unwrap_or(0.0),
                    fov: result.fov.unwrap_or(0.0),
                    distortion: result.distortion,
                    rmse: result.rmse.unwrap_or(0.0),
                    p90_error: result.p90e.unwrap_or(0.0),
                    solve_time: solve_duration.try_into().ok(),
                    ..Default::default()
                })
            }
            SolveStatus::NoMatch => Err(not_found_error("No matches found in database")),
            SolveStatus::Timeout => Err(deadline_exceeded_error("Solve timed out")),
            SolveStatus::Cancelled => Err(deadline_exceeded_error("Solve was cancelled")),
            SolveStatus::TooFew => Err(invalid_argument_error("Too few stars detected")),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::Read,
        path::Path,
        time::Instant,
    };

    use zip::ZipArchive;

    use tetra3::{SolveOptions, SolveStatus, Tetra3};

    /// Minimal proto wire-format parser: extracts centroids, image dimensions,
    /// fov_estimate, and fov_max_error from serialized SolveRequest without
    /// requiring a matching .proto schema.
    struct ParsedRequest {
        centroids: Vec<[f64; 2]>,
        image_height: i32,
        image_width: i32,
        fov_estimate: Option<f64>,
        fov_max_error: Option<f64>,
    }

    fn parse_solve_request(data: &[u8]) -> ParsedRequest {
        let mut centroids = Vec::new();
        let mut image_height: i32 = 0;
        let mut image_width: i32 = 0;
        let mut fov_estimate = None;
        let mut fov_max_error = None;

        let mut i = 0;
        while i < data.len() {
            let (tag, new_i) = read_varint(data, i);
            i = new_i;
            let field_num = tag >> 3;
            let wire_type = tag & 0x07;

            match wire_type {
                0 => {
                    let (val, new_i) = read_varint(data, i);
                    i = new_i;
                    // In the original proto: field 2 = image_width, field 3 = image_height
                    match field_num {
                        2 => image_width = val as i32,
                        3 => image_height = val as i32,
                        _ => {}
                    }
                }
                1 => {
                    let val = f64::from_le_bytes(data[i..i + 8].try_into().unwrap());
                    i += 8;
                    match field_num {
                        4 => fov_estimate = Some(val),
                        5 => fov_max_error = Some(val),
                        _ => {}
                    }
                }
                2 => {
                    let (length, new_i) = read_varint(data, i);
                    i = new_i;
                    let sub_data = &data[i..i + length as usize];
                    i += length as usize;
                    if field_num == 1 {
                        let (x, y) = parse_image_coord(sub_data);
                        centroids.push([x, y]);
                    }
                }
                5 => { i += 4; }
                _ => break,
            }
        }

        ParsedRequest { centroids, image_height, image_width, fov_estimate, fov_max_error }
    }

    fn parse_image_coord(data: &[u8]) -> (f64, f64) {
        let mut x = 0.0;
        let mut y = 0.0;
        let mut i = 0;
        while i < data.len() {
            let (tag, new_i) = read_varint(data, i);
            i = new_i;
            let field_num = tag >> 3;
            let wire_type = tag & 0x07;
            if wire_type == 1 {
                let val = f64::from_le_bytes(data[i..i + 8].try_into().unwrap());
                i += 8;
                match field_num {
                    1 => x = val,
                    2 => y = val,
                    _ => {}
                }
            } else {
                break;
            }
        }
        (x, y)
    }

    struct ParsedResult {
        ra: f64,
        dec: f64,
        roll: f64,
        fov: f64,
        is_match: bool,
    }

    fn parse_solve_result(data: &[u8]) -> ParsedResult {
        let mut ra = 0.0;
        let mut dec = 0.0;
        let mut roll = 0.0;
        let mut fov = 0.0;
        let mut status: u64 = 0;
        let mut has_status = false;

        let mut i = 0;
        while i < data.len() {
            let (tag, new_i) = read_varint(data, i);
            i = new_i;
            let field_num = tag >> 3;
            let wire_type = tag & 0x07;

            match wire_type {
                0 => {
                    let (val, new_i) = read_varint(data, i);
                    i = new_i;
                    if field_num == 14 {
                        status = val;
                        has_status = true;
                    }
                }
                1 => {
                    let val = f64::from_le_bytes(data[i..i + 8].try_into().unwrap());
                    i += 8;
                    match field_num {
                        2 => roll = val,
                        3 => fov = val,
                        _ => {}
                    }
                }
                2 => {
                    let (length, new_i) = read_varint(data, i);
                    i = new_i;
                    let sub_data = &data[i..i + length as usize];
                    i += length as usize;
                    if field_num == 1 {
                        let mut si = 0;
                        while si < sub_data.len() {
                            let (stag, sni) = read_varint(sub_data, si);
                            si = sni;
                            let sfn = stag >> 3;
                            let swt = stag & 0x07;
                            if swt == 1 {
                                let val = f64::from_le_bytes(
                                    sub_data[si..si + 8].try_into().unwrap(),
                                );
                                si += 8;
                                match sfn {
                                    1 => ra = val,
                                    2 => dec = val,
                                    _ => {}
                                }
                            } else {
                                break;
                            }
                        }
                    }
                }
                5 => { i += 4; }
                _ => break,
            }
        }

        ParsedResult { ra, dec, roll, fov, is_match: has_status && status == 1 }
    }

    fn read_varint(data: &[u8], mut i: usize) -> (u64, usize) {
        let mut val: u64 = 0;
        let mut shift = 0;
        loop {
            let b = data[i];
            i += 1;
            val |= ((b & 0x7f) as u64) << shift;
            shift += 7;
            if b & 0x80 == 0 { break; }
        }
        (val, i)
    }

    #[tokio::test]
    async fn test_solver_benchmark() {
        let db_path = Path::new("data/default_database.npz");
        if !db_path.exists() {
            eprintln!("Skipping test: default_database.npz not found.");
            return;
        }

        let db_load_start = Instant::now();
        let mut tetra3 =
            Tetra3::load_database(db_path).expect("Failed to load Tetra3 database");
        println!("Database loaded in {:.2?}", db_load_start.elapsed());

        let zip_path = Path::new("data/testdata.zip");
        let zip_file = File::open(zip_path)
            .expect("Failed to open data/testdata.zip. Ensure the file exists.");
        let mut archive = ZipArchive::new(zip_file).expect("Failed to open zip archive");

        let mut all_failures = Vec::new();
        let mut match_count = 0;
        let mut total_solve_micros: u128 = 0;
        let iterations = 738;

        for x in 0..iterations {
            let req_filename = format!("solve_request_{}.pb", x);
            let mut req_buffer = Vec::new();
            {
                let mut req_file = archive
                    .by_name(&req_filename)
                    .unwrap_or_else(|_| panic!("Entry {} not found in zip", req_filename));
                req_file.read_to_end(&mut req_buffer).unwrap();
            }
            let request = parse_solve_request(&req_buffer);

            let res_filename = format!("solve_result_{}.pb", x);
            let mut res_buffer = Vec::new();
            {
                let mut res_file = archive
                    .by_name(&res_filename)
                    .unwrap_or_else(|_| panic!("Entry {} not found in zip", res_filename));
                res_file.read_to_end(&mut res_buffer).unwrap();
            }
            let expected = parse_solve_result(&res_buffer);

            // Tetra3 expects [[y, x], ...]
            let mut centroids_arr =
                ndarray::Array2::<f64>::zeros((request.centroids.len(), 2));
            for (i, coord) in request.centroids.iter().enumerate() {
                centroids_arr[[i, 0]] = coord[1]; // y
                centroids_arr[[i, 1]] = coord[0]; // x
            }

            let mut options = SolveOptions::default();
            options.fov_estimate = request.fov_estimate;
            options.fov_max_error = request.fov_max_error;

            let size = (request.image_height as f64, request.image_width as f64);

            let start_time = Instant::now();
            let result = tetra3.solve_from_centroids(&centroids_arr, size, options);
            let solve_duration = start_time.elapsed();
            total_solve_micros += solve_duration.as_micros();

            if result.status == SolveStatus::MatchFound {
                match_count += 1;
            }

            // Cross-implementation tolerance (expected from Python solver)
            if expected.is_match && result.status == SolveStatus::MatchFound {
                let epsilon = 0.25; // degrees
                let result_ra = result.ra.unwrap_or(0.0);
                let result_dec = result.dec.unwrap_or(0.0);
                let result_roll = result.roll.unwrap_or(0.0);
                let result_fov = result.fov.unwrap_or(0.0);

                let mut sample_errors = Vec::new();
                if (result_ra - expected.ra).abs() >= epsilon {
                    sample_errors.push(format!(
                        "RA: expected {:.6}, got {:.6} (diff {:.6})",
                        expected.ra, result_ra, (result_ra - expected.ra).abs()
                    ));
                }
                if (result_dec - expected.dec).abs() >= epsilon {
                    sample_errors.push(format!(
                        "Dec: expected {:.6}, got {:.6} (diff {:.6})",
                        expected.dec, result_dec, (result_dec - expected.dec).abs()
                    ));
                }
                if (result_roll - expected.roll).abs() >= epsilon {
                    sample_errors.push(format!(
                        "Roll: expected {:.6}, got {:.6} (diff {:.6})",
                        expected.roll, result_roll, (result_roll - expected.roll).abs()
                    ));
                }
                if (result_fov - expected.fov).abs() >= epsilon {
                    sample_errors.push(format!(
                        "FOV: expected {:.6}, got {:.6} (diff {:.6})",
                        expected.fov, result_fov, (result_fov - expected.fov).abs()
                    ));
                }

                if !sample_errors.is_empty() {
                    all_failures.push(format!(
                        "Sample {} failures:\n  {}",
                        x,
                        sample_errors.join("\n  ")
                    ));
                }
            } else if expected.is_match && result.status != SolveStatus::MatchFound {
                all_failures.push(format!(
                    "Sample {}: Expected match but got {:?}",
                    x, result.status
                ));
            }
        }

        let total_ms = total_solve_micros as f64 / 1000.0;
        println!(
            "\n=== Performance Report ===\n\
             Total iterations: {}\n\
             Matches found: {}\n\
             Total solver time: {:.2} ms\n\
             Average time per solve: {:.2} ms\n",
            iterations,
            match_count,
            total_ms,
            total_ms / iterations as f64,
        );

        if !all_failures.is_empty() {
            panic!(
                "{} of {} test samples failed:\n\n{}",
                all_failures.len(),
                iterations,
                all_failures.join("\n\n")
            );
        }
    }
}
