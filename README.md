# Tetra3 Solver in Rust

A fast, robust, and async-friendly Rust implementation of the [cedar-solve](https://github.com/smroid/cedar-solve) plate solving algorithm.

## Repository Structure

This workspace is divided into two primary crates:

* **`tetra3`**: The core plate-solving algorithm. This is a Rust port of the [Tetra3](https://github.com/smroid/cedar-solve/blob/master/tetra3/tetra3.py) `solve_from_centroids` function.
* **`cedar-solver`**: The integration layer for the [Cedar™](https://github.com/smroid/cedar) telescope control system. Also contains the standalone `tetra3-solve-server` gRPC binary.

## Getting Started

### Prerequisites
* [Rust / Cargo](https://rustup.rs/) (edition 2021)
* [protoc](https://grpc.io/docs/protoc-installation/) (Protocol Buffer compiler)
* [cedar-server](https://github.com/smroid/cedar-server) cloned into the same parent directory as tetra-solve-rs
* [tetra3_server](https://github.com/smroid/tetra3_server) cloned into the same parent directory (needed for the proto definition)
* Python 3 (Optional, for running the Python test suite)

### Building

Build everything (libraries + server binary):

```
cargo build --release
```

Build only the libraries (no server binary):

```
cargo build --release --lib -p tetra3
cargo build --release --lib -p cedar-solver
```

Build only the server binary:

```
cargo build --release --bin tetra3-solve-server
```

The server binary is output to `target/release/tetra3-solve-server`.

## Standalone Server

The `tetra3-solve-server` binary is a gRPC server that exposes the `SolveFromCentroids` RPC. It is designed to run alongside [cedar-detect](https://github.com/smroid/cedar-detect) as part of a plate-solving pipeline:

```
Image -> cedar-detect (:50051) -> star centroids -> tetra3-solve-server (:50052) -> plate solution
```

### Deployment

The binary is self-contained (only links to libc). To deploy, copy two files to the target machine:

1. `target/release/tetra3-solve-server` (the binary)
2. `cedar-solver/data/default_database.npz` (the star catalog, 14.6 MB)

### Usage

```
tetra3-solve-server --database /path/to/default_database.npz [--port 50052]
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `-d, --database` | (required) | Path to the Tetra3 `.npz` star catalog |
| `-p, --port` | `50052` | gRPC listen port |

Control log verbosity with the `RUST_LOG` environment variable:

```
RUST_LOG=debug tetra3-solve-server --database /path/to/default_database.npz
```

### Running as a systemd Service

A sample unit file is provided at `tetra3_solve_server.service`. Edit the `ExecStart` path and `User` to match your deployment, then:

```
sudo cp tetra3_solve_server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tetra3_solve_server
sudo systemctl start tetra3_solve_server
```

### gRPC API

The server implements the `Tetra3` service defined in [tetra3.proto](https://github.com/smroid/tetra3_server/blob/master/proto/tetra3.proto):

```protobuf
service Tetra3 {
  rpc SolveFromCentroids(SolveRequest) returns (SolveResult);
}
```

Any gRPC client can call the endpoint. Example with Python:

```python
import grpc
from tetra3_server import tetra3_pb2, tetra3_pb2_grpc

channel = grpc.insecure_channel('localhost:50052')
stub = tetra3_pb2_grpc.Tetra3Stub(channel)

request = tetra3_pb2.SolveRequest(
    star_centroids=[tetra3_pb2.ImageCoord(x=100.5, y=200.3), ...],
    image_width=1280,
    image_height=960,
    fov_estimate=11.0,
    fov_max_error=3.0,
)
result = stub.SolveFromCentroids(request)
print(result.image_center_coords.ra, result.image_center_coords.dec)
```

### Port Allocation

| Service | Default Port |
|---------|-------------|
| cedar-detect | 50051 |
| tetra3-solve-server | 50052 |

## Testing

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
