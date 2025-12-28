// Copyright (c) 2025 Omair Kamil oakamil@gmail.com
// See LICENSE file in root directory for license terms.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering as AtomicOrdering},
        Arc,
    },
    time::Duration,
};

use async_trait::async_trait;
use canonical_error::{
    deadline_exceeded_error, invalid_argument_error, not_found_error,
    CanonicalError,
};
use cedar_elements::{
    cedar::{ImageCoord, PlateSolution},
    cedar_common::CelestialCoord,
    imu_trait::EquatorialCoordinates,
    solver_trait::{SolveExtension, SolveParams, SolverTrait},
};
use ndarray::Array2;

use crate::tetra3::Tetra3;

// Status constants matching Tetra3 implementation
const MATCH_FOUND: u8 = 1;
const NO_MATCH: u8 = 2;
const TIMEOUT: u8 = 3;
const CANCELLED: u8 = 4;
const TOO_FEW: u8 = 5;

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

        // Map SolveParams to Tetra3 arguments
        let fov_estimate = params.fov_estimate.map(|(fov, _)| fov);
        let fov_max_error = params.fov_estimate.map(|(_, err)| err);
        let solve_timeout_ms =
            params.solve_timeout.map(|d| d.as_millis() as u64);

        // Synchronize trait cancellation flag with the Tetra3 instance
        // (Note: This assumes Tetra3 has been updated to check this flag)
        inner.set_cancelled(false);

        let result = inner.solve_from_centroids(
            &centroids_arr,
            (height as u32, width as u32),
            fov_estimate,
            fov_max_error,
            params.match_radius,
            params.match_threshold,
            solve_timeout_ms,
            params.distortion,
        );

        match result.status {
            MATCH_FOUND => {
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
                    seconds: (result.t_solve / 1000.0) as i64,
                    nanos: ((result.t_solve % 1000.0) * 1_000_000.0) as i32,
                });

                // Note: rotation_matrix, target_pixel, and target_sky_coord are
                // omitted here as the provided SolveResult does
                // not include the raw rotation matrix,
                // and coordinate transforms are typically handled by Cedar's
                // astro_util.rs.

                Ok(plate_solution)
            }
            NO_MATCH => Err(not_found_error("Solver failed to find a match.")),
            TIMEOUT => Err(deadline_exceeded_error("Solver timed out.")),
            CANCELLED => {
                Err(deadline_exceeded_error("Solve operation was cancelled."))
            }
            TOO_FEW => Err(invalid_argument_error(
                "Too few centroids to attempt solve.",
            )),
            _ => Err(not_found_error(
                "Solver encountered an unknown error state.",
            )),
        }
    }

    fn cancel(&self) {
        // Atomically set the flag to signal the solver loop to terminate
        self.cancelled.store(true, AtomicOrdering::SeqCst);

        // Attempt to set the flag on the inner instance if the lock is
        // available
        if let Ok(mut inner) = self.inner.try_lock() {
            inner.set_cancelled(true);
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
    };

    use prost::Message; /* Ensure 'prost' is in your Cargo.toml
                          * dependencies */
    use zip::ZipArchive;

    use super::*;
    // Import the necessary structs from your crate
    // Adjust 'crate::tetra3' and 'crate::tetra3_server' to match your
    // project structure
    use crate::tetra3::Tetra3;
    use crate::{
        tetra3_server::{SolveRequest, SolveResult, SolveStatus},
        tetra3_solver::Tetra3Solver,
    };

    #[tokio::test]
    async fn test_solver_consistency_with_testdata() {
        // 1. Initialize the Tetra3Solver
        // We assume the default database is available in the root directory.
        // Update this path if your database is located elsewhere (e.g.,
        // "data/default_database.npz").
        let db_path = Path::new("test/default_database.npz");
        if !db_path.exists() {
            eprintln!("Skipping test: default_database.npz not found.");
            return;
        }
        let solver = Tetra3Solver::new(
            Tetra3::new(db_path.to_str())
                .expect("Failed to load Tetra3 database"),
        );

        // 2. Open the testdata.zip file
        let zip_path = Path::new("test/testdata.zip");
        let zip_file = File::open(zip_path).expect(
            "Failed to open test/testdata.zip. Ensure the file exists.",
        );
        let mut archive =
            ZipArchive::new(zip_file).expect("Failed to open zip archive");

        // 3. Iterate from x = 0 to 737
        for x in 0..=737 {
            // --- Parse SolveRequest ---
            let req_filename = format!("solve_request_{}.pb", x);
            let mut req_buffer = Vec::new();

            {
                let mut req_file =
                    archive.by_name(&req_filename).unwrap_or_else(|_| {
                        panic!("Entry {} not found in zip", req_filename)
                    });
                req_file.read_to_end(&mut req_buffer).unwrap();
            }

            let request = SolveRequest::decode(Cursor::new(req_buffer))
                .expect("Failed to decode SolveRequest proto");

            // --- Parse SolveResult (Expected) ---
            let res_filename = format!("solve_result_{}.pb", x);
            let mut res_file =
                archive.by_name(&res_filename).unwrap_or_else(|_| {
                    panic!("Entry {} not found in zip", res_filename)
                });

            let mut res_buffer = Vec::new();
            res_file.read_to_end(&mut res_buffer).unwrap();

            let expected_result = SolveResult::decode(Cursor::new(res_buffer))
                .expect("Failed to decode SolveResult proto");

            // --- Invoke solve_from_centroids ---
            // We extract the relevant fields from the request to pass to the
            // solver. Note: If your solve_from_centroids accepts
            // the 'SolveRequest' struct directly, you can pass
            // 'request' instead of extracting individual fields.

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

            let res = solver
                .solve_from_centroids(
                    &centroids, width, height, &extension, &params, None,
                )
                .await;

            match res {
                Ok(ref _s) => {}
                Err(e) => {
                    if expected_result.status
                        == Some(SolveStatus::MatchFound as i32)
                    {
                        panic!("Expected a valid solution for iteration {} but found error: {:?}", x, e);
                    }
                    continue;
                }
            };
            let result = res.unwrap();

            // We verify RA, Dec, Roll, and FOV are within a reasonable
            // tolerance.
            if expected_result.status == Some(SolveStatus::MatchFound as i32) {
                let epsilon = 1e-5; // Tolerance for floating point comparison
                let expected_coords =
                    expected_result.image_center_coords.unwrap();
                let expected_ra = expected_coords.ra;
                let expected_dec = expected_coords.dec;
                let expected_roll = expected_result.roll.unwrap();
                let expected_fov = expected_result.roll.unwrap();
                let result_coord = result.image_sky_coord.unwrap();
                let result_ra = result_coord.ra;
                let result_dec = result_coord.dec;

                assert!(
                    (result_ra - expected_ra).abs() < epsilon,
                    "RA mismatch at sample {}: expected {}, got {}",
                    x,
                    expected_ra,
                    result_ra
                );

                assert!(
                    (result_dec - expected_dec).abs() < epsilon,
                    "Dec mismatch at sample {}: expected {}, got {}",
                    x,
                    expected_dec,
                    result_dec
                );

                assert!(
                    (result.roll - expected_roll).abs() < epsilon,
                    "Roll mismatch at sample {}: expected {}, got {}",
                    x,
                    expected_roll,
                    result.roll
                );

                assert!(
                    (result.fov - expected_fov).abs() < epsilon,
                    "FOV mismatch at sample {}: expected {}, got {}",
                    x,
                    expected_fov,
                    result.fov
                );
            }
        }
    }
}
