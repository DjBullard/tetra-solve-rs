// Copyright (c) 2026 Omair Kamil oakamil@gmail.com
// See LICENSE file in root directory for license terms.

use std::net::SocketAddr;
use std::path::Path;
use std::time::{Duration as StdDuration, Instant};

use clap::Parser;
use log::info;
use ndarray::Array2;
use tonic_web::GrpcWebLayer;

use tetra3::{SolveOptions, SolveStatus, Tetra3};

pub mod tetra3_server {
    tonic::include_proto!("tetra3_server");
}

use tetra3_server::tetra3_server::{Tetra3 as Tetra3Rpc, Tetra3Server};
use tetra3_server::{
    CelestialCoord as ProtoCelestialCoord, ImageCoord as ProtoImageCoord,
    MatchedStar as ProtoMatchedStar, RotationMatrix as ProtoRotationMatrix, SolveRequest,
    SolveResult, SolveStatus as ProtoSolveStatus,
};

struct MyTetra3Solver {
    solver: tokio::sync::Mutex<Tetra3>,
}

#[tonic::async_trait]
impl Tetra3Rpc for MyTetra3Solver {
    async fn solve_from_centroids(
        &self,
        request: tonic::Request<SolveRequest>,
    ) -> Result<tonic::Response<SolveResult>, tonic::Status> {
        let rpc_start = Instant::now();
        let req = request.into_inner();

        // Convert star_centroids: proto ImageCoord (x, y) -> Array2 with (y, x) ordering.
        let mut centroids_arr = Array2::<f64>::zeros((req.star_centroids.len(), 2));
        for (i, coord) in req.star_centroids.iter().enumerate() {
            centroids_arr[[i, 0]] = coord.y;
            centroids_arr[[i, 1]] = coord.x;
        }

        let size = (req.image_height as f64, req.image_width as f64);

        // Build SolveOptions from request fields.
        let default_options = SolveOptions::default();
        let mut options = SolveOptions {
            fov_estimate: req.fov_estimate,
            fov_max_error: req.fov_max_error,
            match_radius: req.match_radius.unwrap_or(default_options.match_radius),
            match_threshold: req.match_threshold.unwrap_or(default_options.match_threshold),
            solve_timeout_ms: req.solve_timeout.as_ref().map(|d| {
                d.seconds as f64 * 1000.0 + d.nanos as f64 / 1_000_000.0
            }),
            distortion: req.distortion,
            match_max_error: req.match_max_error.unwrap_or(default_options.match_max_error),
            return_matches: req.return_matches,
            return_catalog: req.return_catalog,
            return_rotation_matrix: req.return_rotation_matrix,
            ..default_options
        };

        // Map target_pixels (y,x swap).
        if !req.target_pixels.is_empty() {
            let mut target_px_arr = Array2::zeros((req.target_pixels.len(), 2));
            for (i, coord) in req.target_pixels.iter().enumerate() {
                target_px_arr[[i, 0]] = coord.y;
                target_px_arr[[i, 1]] = coord.x;
            }
            options.target_pixel = Some(target_px_arr);
        }

        // Map target_sky_coords (ra, dec -- no swap needed).
        if !req.target_sky_coords.is_empty() {
            let mut target_sky_arr = Array2::zeros((req.target_sky_coords.len(), 2));
            for (i, coord) in req.target_sky_coords.iter().enumerate() {
                target_sky_arr[[i, 0]] = coord.ra;
                target_sky_arr[[i, 1]] = coord.dec;
            }
            options.target_sky_coord = Some(target_sky_arr);
        }

        // Hard gRPC-level timeout backstop. Catches mutex contention and
        // any edge case where the solver overshoots its internal checks.
        let timeout_duration = req.solve_timeout
            .map(|d| {
                StdDuration::from_secs(d.seconds as u64)
                    + StdDuration::from_nanos(d.nanos as u64)
            })
            .unwrap_or(StdDuration::from_secs(5));

        let solver_outcome = tokio::time::timeout(timeout_duration, async {
            let mut solver = self.solver.lock().await;
            solver.solve_from_centroids(&centroids_arr, size, options)
        })
        .await;

        // Map Solution -> proto SolveResult.
        let solve_time = prost_types::Duration::try_from(rpc_start.elapsed()).ok();

        let result = match solver_outcome {
            Ok(r) => r,
            Err(_elapsed) => {
                // tokio timeout fired — return Timeout status.
                return Ok(tonic::Response::new(SolveResult {
                    status: Some(ProtoSolveStatus::Timeout.into()),
                    solve_time,
                    ..Default::default()
                }));
            }
        };

        let status = match result.status {
            SolveStatus::MatchFound => ProtoSolveStatus::MatchFound,
            SolveStatus::NoMatch => ProtoSolveStatus::NoMatch,
            SolveStatus::Timeout => ProtoSolveStatus::Timeout,
            SolveStatus::Cancelled => ProtoSolveStatus::Cancelled,
            SolveStatus::TooFew => ProtoSolveStatus::TooFew,
        };

        let mut solve_result = SolveResult {
            status: Some(status.into()),
            solve_time,
            ..Default::default()
        };

        if result.status == SolveStatus::MatchFound {
            if let (Some(ra), Some(dec)) = (result.ra, result.dec) {
                solve_result.image_center_coords = Some(ProtoCelestialCoord { ra, dec });
            }
            solve_result.roll = result.roll;
            solve_result.fov = result.fov;
            solve_result.distortion = result.distortion;
            solve_result.rmse = result.rmse;
            solve_result.p90e = result.p90e;
            solve_result.maxe = result.maxe;
            solve_result.matches = result.matches.map(|m| m as i32);
            solve_result.prob = result.prob;
            solve_result.epoch_equinox = result.epoch_equinox;
            solve_result.epoch_proper_motion = result.epoch_proper_motion;

            // target_coords: celestial coordinates of request target_pixels.
            if let (Some(target_ra), Some(target_dec)) = (&result.target_ra, &result.target_dec) {
                for (ra, dec) in target_ra.iter().zip(target_dec.iter()) {
                    solve_result
                        .target_coords
                        .push(ProtoCelestialCoord { ra: *ra, dec: *dec });
                }
            }

            // target_sky_to_image_coords: image coords of request target_sky_coords.
            // Out-of-FOV entries are None -> mapped to (-1, -1) per proto spec.
            if let (Some(target_y), Some(target_x)) = (&result.target_y, &result.target_x) {
                for (oy, ox) in target_y.iter().zip(target_x.iter()) {
                    let (x, y) = match (ox, oy) {
                        (Some(x_val), Some(y_val)) => (*x_val, *y_val),
                        _ => (-1.0, -1.0),
                    };
                    solve_result
                        .target_sky_to_image_coords
                        .push(ProtoImageCoord { x, y });
                }
            }

            // matched_stars: centroids are [y, x], stars are [ra, dec, mag].
            if let (Some(stars), Some(centroids)) =
                (&result.matched_stars, &result.matched_centroids)
            {
                let cat_ids = result.matched_cat_id.as_ref();
                for (idx, (star, centroid)) in stars.iter().zip(centroids.iter()).enumerate() {
                    let cat_id = cat_ids
                        .and_then(|ids| ids.get(idx))
                        .map(|id| id.to_string());
                    solve_result.matched_stars.push(ProtoMatchedStar {
                        celestial_coord: Some(ProtoCelestialCoord {
                            ra: star[0],
                            dec: star[1],
                        }),
                        magnitude: star[2],
                        image_coord: Some(ProtoImageCoord {
                            x: centroid[1],
                            y: centroid[0],
                        }),
                        cat_id,
                    });
                }
            }

            // catalog_stars: (ra, dec, mag, y, x) -- cat_id omitted per proto.
            if let Some(cat_stars) = &result.catalog_stars {
                for &(ra, dec, mag, y, x) in cat_stars {
                    solve_result.catalog_stars.push(ProtoMatchedStar {
                        celestial_coord: Some(ProtoCelestialCoord { ra, dec }),
                        magnitude: mag,
                        image_coord: Some(ProtoImageCoord { x, y }),
                        cat_id: None,
                    });
                }
            }

            // rotation_matrix: 3x3 in row-major order.
            if let Some(rm) = &result.rotation_matrix {
                let elements: Vec<f64> = rm.iter().cloned().collect();
                solve_result.rotation_matrix = Some(ProtoRotationMatrix {
                    matrix_elements: elements,
                });
            }
        }

        Ok(tonic::Response::new(solve_result))
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Tetra3 plate solving gRPC server", long_about = None)]
struct Args {
    /// Port that the gRPC server listens on.
    #[arg(short, long, default_value_t = 50052)]
    port: u16,

    /// Path to the Tetra3 .npz database file.
    #[arg(short, long)]
    database: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    #[cfg(target_os = "linux")]
    {
        prctl::set_death_signal(15).unwrap();
    }

    let db_path = Path::new(&args.database);
    info!("Loading Tetra3 database from {:?}...", db_path);
    let tetra3 = Tetra3::load_database(db_path)
        .map_err(|e| format!("Failed to load database: {}", e))?;
    info!("Database loaded successfully.");

    let service = MyTetra3Solver {
        solver: tokio::sync::Mutex::new(tetra3),
    };

    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));
    info!("Tetra3SolveServer listening on {}", addr);

    tonic::transport::Server::builder()
        .accept_http1(true)
        .layer(GrpcWebLayer::new())
        .add_service(Tetra3Server::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
