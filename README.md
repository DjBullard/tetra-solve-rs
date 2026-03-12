# Tetra3 Solver in Rust

A fast, robust, and async-friendly Rust implementation of the [cedar-solve](https://github.com/smroid/cedar-solve) plate solving algorithm. 

## Repository Structure

This workspace is divided into two primary crates:

* **`tetra3`**: The core plate-solving algorithm. This is a Rust port of the [Tetra3](https://github.com/smroid/cedar-solve/blob/master/tetra3/tetra3.py) `solve_from_centroids` function.
* **`cedar-solver`**: The integration layer for the [Cedar™](https://github.com/smroid/cedar) telescope control system.

## Getting Started

### Prerequisites
* [Rust / Cargo](https://rustup.rs/) (edition 2021)
* Python 3 (Optional, for running the Python test suite)
* [cedar-server](https://github.com/smroid/cedar-server) cloned into the same location as tetra-solve-rs

### Building
To build the workspace:

```
cargo build --release
```

### Testing

A set of real-world test data is provided for validating the algorithm. The tests are located in the `cedar-solver` crate.

#### Python Tetra3 Validation

To run the Python tests, ensure that you have the following repos cloned into the same location as tetra-solve-rs:
* [cedar-solve](https://github.com/smroid/cedar-solve)
* [tetra3_server](https://github.com/smroid/tetra3_server)

In `cedar-solve` ensure that the `setup.sh` script is run. Then run the tests against the Python solver:

```
./run_python_tets.sh
```

#### Rust Port Validation

Ensure that `tetra3-server` is cloned to the same location as `tetra-solve-rs`.

```
cargo test --release tetra3_solver -- --nocapture
```

## FAQ

1\. Why port only the solving function?

A Rust implementation of the star detection algorithm is already available in the [cedar-detect](https://github.com/smroid/cedar-detect) repo. Database generation is a one-time operation that doesn't benefit from a port.

2\. What kind of performance gain can I expect to see?

This depends on the hardware. On a Raspberry Pi 5 with 4 GB RAM the Rust version is only ~20% faster. On a Raspberry Pi Zero 2W with 512 MB of RAM the Rust version is >10x faster.

## License

This project is licensed under the Functional Source License, Version 1.1, MIT Future License (FSL-1.1-MIT).

See LICENSE.md for full details.

## Disclaimer

All product names, trademarks and registered trademarks are property of their respective owners. All company, product and service names used in this website are for identification purposes only. Use of these names, trademarks and brands does not imply endorsement.

`tetra-solve-rs` is not affiliated with, endorsed by, or sponsored by Clear Skies Astro.

Cedar™ is a trademark of Clear Skies Astro, registered in the U.S. and other countries.
