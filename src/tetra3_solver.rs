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

use crate::tetra3::{SolveOptions, SolveStatus, Tetra3};

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
    async fn solve_from_centroids(
        &self,
        star_centroids: &[ImageCoord],
        width: usize,
        height: usize,
        extension: &SolveExtension,
        params: &SolveParams,
        _imu_estimate: Option<EquatorialCoordinates>,
    ) -> Result<PlateSolution, CanonicalError> {
        // Reset cancellation state before starting a new solve
        self.cancelled.store(false, AtomicOrdering::SeqCst);

        let mut inner = self.inner.lock().await;

        // Map ImageCoord slice to ndarray Array2 (N x 2)
        // Tetra3 expects [[y, x], ...] as per its internal processing logic
        let mut centroids_arr = Array2::<f64>::zeros((star_centroids.len(), 2));
        for (i, coord) in star_centroids.iter().enumerate() {
            centroids_arr[[i, 0]] = coord.y;
            centroids_arr[[i, 1]] = coord.x;
        }

        // Fetch Tetra3 defaults to use as fallbacks for optional parameters
        let default_options = SolveOptions::default();

        // Map SolveParams and SolveExtension to Tetra3 SolveOptions
        let mut options = SolveOptions {
            fov_estimate: params.fov_estimate.map(|(fov, _)| fov),
            fov_max_error: params.fov_estimate.map(|(_, err)| err),
            match_radius: params.match_radius.unwrap_or(default_options.match_radius),
            match_threshold: params
                .match_threshold
                .unwrap_or(default_options.match_threshold),
            solve_timeout_ms: params.solve_timeout.map(|d| d.as_millis() as f64),
            distortion: params.distortion,
            match_max_error: params
                .match_max_error
                .unwrap_or(default_options.match_max_error),
            ..default_options
        };

        // Pass target pixels if requested via extension
        if let Some(tp) = &extension.target_pixel {
            let mut target_px_arr = Array2::zeros((tp.len(), 2));
            for (i, coord) in tp.iter().enumerate() {
                target_px_arr[[i, 0]] = coord.y;
                target_px_arr[[i, 1]] = coord.x;
            }
            options.target_pixel = Some(target_px_arr);
        }

        // Pass target sky coordinates if requested via extension
        if let Some(tsc) = &extension.target_sky_coord {
            let mut target_sky_arr = Array2::zeros((tsc.len(), 2));
            for (i, coord) in tsc.iter().enumerate() {
                target_sky_arr[[i, 0]] = coord.ra;
                target_sky_arr[[i, 1]] = coord.dec;
            }
            options.target_sky_coord = Some(target_sky_arr);
        }

        // Check if the overall task was cancelled prior to the lock acquisition
        if self.cancelled.load(AtomicOrdering::SeqCst) {
            return Err(deadline_exceeded_error("Solve operation was cancelled."));
        }

        let result =
            inner.solve_from_centroids(&centroids_arr, (height as f64, width as f64), options);

        match result.status {
            SolveStatus::MatchFound => {
                let mut plate_solution = PlateSolution::default();

                // Populate core coordinates
                plate_solution.image_sky_coord = Some(CelestialCoord {
                    ra: result.ra.unwrap_or(0.0),
                    dec: result.dec.unwrap_or(0.0),
                });
                plate_solution.roll = result.roll.unwrap_or(0.0);
                plate_solution.fov = result.fov.unwrap_or(0.0);
                plate_solution.distortion = result.distortion;

                // Populate quality metrics
                plate_solution.rmse = result.rmse.unwrap_or(0.0);
                plate_solution.num_matches = result.matches.unwrap_or(0) as i32;
                plate_solution.prob = result.prob.unwrap_or(0.0);

                // Populate timing information (convert ms to proto Duration)
                plate_solution.solve_time = Some(prost_types::Duration {
                    seconds: (result.t_solve_ms / 1000.0) as i64,
                    nanos: ((result.t_solve_ms % 1000.0) * 1_000_000.0) as i32,
                });

                Ok(plate_solution)
            }
            SolveStatus::NoMatch => Err(not_found_error("Solver failed to find a match.")),
            SolveStatus::Timeout => Err(deadline_exceeded_error("Solver timed out.")),
            SolveStatus::Cancelled => {
                Err(deadline_exceeded_error("Solve operation was cancelled."))
            }
            SolveStatus::TooFew => Err(invalid_argument_error(
                "Too few centroids to attempt solve.",
            )),
        }
    }

    fn cancel(&self) {
        // Atomically set the flag to signal the solver loop to terminate
        self.cancelled.store(true, AtomicOrdering::SeqCst);

        // Attempt to set the flag on the inner instance if the lock is available
        if let Ok(mut inner) = self.inner.try_lock() {
            inner.cancel_solve();
        }
    }

    fn default_timeout(&self) -> Duration {
        // Return a default solve duration if not specified by params
        Duration::from_secs(1)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{Cursor, Read},
        path::Path,
        time::Instant,
    };

    use prost::Message;
    use zip::ZipArchive;

    use super::*;
    use crate::tetra3::Tetra3;
    use crate::{
        tetra3_server::{SolveRequest, SolveResult, SolveStatus as ProtoSolveStatus},
        tetra3_solver::Tetra3Solver,
    };

    #[tokio::test]
    async fn test_solver_consistency_with_testdata() {
        let db_path = Path::new("data/default_database.npz");
        if !db_path.exists() {
            eprintln!("Skipping test: default_database.npz not found.");
            return;
        }
        let solver = Tetra3Solver::new(
            Tetra3::load_database(db_path).expect("Failed to load Tetra3 database"),
        );

        let zip_path = Path::new("data/testdata.zip");
        let zip_file = File::open(zip_path)
            .expect("Failed to open test/testdata.zip. Ensure the file exists.");
        let mut archive = ZipArchive::new(zip_file).expect("Failed to open zip archive");

        // We will collect errors here instead of panicking immediately
        let mut all_failures = Vec::new();
        let mut total_solve_micros = 0;
        let iterations = 738;

        for x in 0..=iterations - 1 {
            let req_filename = format!("solve_request_{}.pb", x);
            let mut req_buffer = Vec::new();

            {
                let mut req_file = archive
                    .by_name(&req_filename)
                    .unwrap_or_else(|_| panic!("Entry {} not found in zip", req_filename));
                req_file.read_to_end(&mut req_buffer).unwrap();
            }

            let request = SolveRequest::decode(Cursor::new(req_buffer))
                .expect("Failed to decode SolveRequest proto");

            let res_filename = format!("solve_result_{}.pb", x);
            let mut res_file = archive
                .by_name(&res_filename)
                .unwrap_or_else(|_| panic!("Entry {} not found in zip", res_filename));

            let mut res_buffer = Vec::new();
            res_file.read_to_end(&mut res_buffer).unwrap();

            let expected_result = SolveResult::decode(Cursor::new(res_buffer))
                .expect("Failed to decode SolveResult proto");

            let centroids: Vec<ImageCoord> = request
                .star_centroids
                .iter()
                .map(|coord| ImageCoord {
                    x: coord.x,
                    y: coord.y,
                })
                .collect();

            let width = request.image_width as usize;
            let height = request.image_height as usize;

            let mut extension = SolveExtension::default();
            let target_pixels: Vec<ImageCoord> = request
                .target_pixels
                .iter()
                .map(|coord| ImageCoord {
                    x: coord.x,
                    y: coord.y,
                })
                .collect();
            extension.target_pixel = Some(target_pixels);

            let target_sky_coords: Vec<CelestialCoord> = request
                .target_sky_coords
                .iter()
                .map(|coord| CelestialCoord {
                    ra: coord.ra,
                    dec: coord.dec,
                })
                .collect();
            extension.target_sky_coord = Some(target_sky_coords);

            let mut params = SolveParams::default();
            params.match_radius = request.match_radius;
            params.match_threshold = request.match_threshold;
            if let (Some(fov_estimate), Some(fov_max_error)) =
                (request.fov_estimate, request.fov_max_error)
            {
                params.fov_estimate = Some((fov_estimate, fov_max_error));
            }
            params.distortion = request.distortion;
            params.match_max_error = request.match_max_error;

            // --- Capture the execution time ---
            let start_time = Instant::now();

            let res = solver
                .solve_from_centroids(&centroids, width, height, &extension, &params, None)
                .await;

            let solve_duration = start_time.elapsed();
            total_solve_micros += solve_duration.as_micros();
            // -----------------------------------

            match res {
                Ok(ref _s) => {}
                Err(e) => {
                    if expected_result.status == Some(ProtoSolveStatus::MatchFound as i32) {
                        all_failures.push(format!(
                            "Sample {}: Expected MatchFound but got error {:?}",
                            x, e
                        ));
                    }
                    continue;
                }
            };
            let result = res.unwrap();

            if expected_result.status == Some(ProtoSolveStatus::MatchFound as i32) {
                let epsilon = 1e-5;
                let expected_coords = expected_result.image_center_coords.unwrap();
                let expected_ra = expected_coords.ra;
                let expected_dec = expected_coords.dec;
                let expected_roll = expected_result.roll.unwrap();
                let expected_fov = expected_result.fov.unwrap();

                let result_coord = result.image_sky_coord.unwrap();
                let result_ra = result_coord.ra;
                let result_dec = result_coord.dec;

                println!("--- Sample {} ---", x);
                println!("Solve time : {:.2?}", solve_duration);
                println!(
                    "Expected   : RA: {:.6}, Dec: {:.6}, Roll: {:.6}, FOV: {:.6}",
                    expected_ra, expected_dec, expected_roll, expected_fov
                );
                println!(
                    "Actual     : RA: {:.6}, Dec: {:.6}, Roll: {:.6}, FOV: {:.6}",
                    result_ra, result_dec, result.roll, result.fov
                );
                println!(
                    "Diff       : RA: {:.6}, Dec: {:.6}, Roll: {:.6}, FOV: {:.6}",
                    (result_ra - expected_ra).abs(),
                    (result_dec - expected_dec).abs(),
                    (result.roll - expected_roll).abs(),
                    (result.fov - expected_fov).abs()
                );
                println!("-------------------\n");

                // Instead of assert!, we push errors to our collection
                let mut sample_errors = Vec::new();

                if (result_ra - expected_ra).abs() >= epsilon {
                    sample_errors.push(format!(
                        "RA mismatch: expected {}, got {}",
                        expected_ra, result_ra
                    ));
                }
                if (result_dec - expected_dec).abs() >= epsilon {
                    sample_errors.push(format!(
                        "Dec mismatch: expected {}, got {}",
                        expected_dec, result_dec
                    ));
                }
                if (result.roll - expected_roll).abs() >= epsilon {
                    sample_errors.push(format!(
                        "Roll mismatch: expected {}, got {}",
                        expected_roll, result.roll
                    ));
                }
                if (result.fov - expected_fov).abs() >= epsilon {
                    sample_errors.push(format!(
                        "FOV mismatch: expected {}, got {}",
                        expected_fov, result.fov
                    ));
                }

                if !sample_errors.is_empty() {
                    all_failures.push(format!(
                        "Sample {} failures:\n  {}",
                        x,
                        sample_errors.join("\n  ")
                    ));
                }
            }
        }

        println!(
            "\n=== Performance Report ===\n\
             Total iterations: {}\n\
             Successful matches: {}\n\
             Pure solver time: {:.2} ms\n\
             Average time per solve: {:.2} ms\n",
            iterations,
            iterations - all_failures.len(),
            total_solve_micros as f64 / 1000.0,
            total_solve_micros as f64 / 1000.0 / (iterations as f64),
        );

        // Finally, panic if there were any failures accumulated across ALL 738 iterations.
        if !all_failures.is_empty() {
            panic!(
                "{} of 738 test samples failed:\n\n{}",
                all_failures.len(),
                all_failures.join("\n\n")
            );
        }
    }
}
