# TensorGator

TensorGator is a CUDA-accelerated satellite propagation library designed for massively parallel orbital mechanics calculations.

## Performance

TensorGator's CUDA backend provides significant performance improvements over CPU-based propagation:

- **Large Constellations (1000+ satellites)**:  Using batch sizes: 500,000 satellites, 500 timesteps in ~21 seconds. Tested on Google Colab T4 GPU (15GB VRAM)

## Features

- **CUDA Acceleration**: Propagate thousands of satellites simultaneously using GPU parallelization
- **J2 Perturbation Model**: Accurate orbital propagation including Earth's oblateness effects
- **Flexible Input Formats**: Support for both Keplerian elements and R,V vectors
- **Coordinate Transformations**: Fast ECI/ECEF conversions with Numba acceleration
- **Satellite Visibility Analysis**: Determine satellite coverage and visibility from the ground
- **Batch Processing**: Memory-efficient handling of large constellations through automatic batching
- **CPU Fall Back**: Support for CPU mode when CUDA is not available

## Installation

```bash
pip install tensorgator
```

Or install from source:

```bash
git clone https://github.com/yourusername/tensorgator.git
cd tensorgator
pip install -e .
```

## Requirements

- Python 3.6+
- CUDA-compatible GPU
- NumPy
- Numba

## Visualization Libraraies
- Matplotlib
- Basemap

## Quick Start


## Core Functions

### Propagation

```python
tg.satellite_positions(times, constellation, backend='cpu', return_frame='ecef', epochs=None, input_type='kepler')
```

Propagates satellite positions over time using either CPU or CUDA backend.

Parameters:
- `times`: Array of times (seconds since J2000 or reference epoch)
- `constellation`: Array of satellite elements (Keplerian or position-velocity)
- `backend`: 'cpu' or 'cuda'
- `return_frame`: Coordinate frame to return ('ecef' or 'eci')
- `epochs`: Optional array of epoch times for each satellite
- `input_type`: 'kepler' for Keplerian elements or 'rv' for position-velocity vectors

### Visibility Analysis

```python
tg.calculate_visibility(satellite_positions, ground_stations, min_elevation=10.0)
```

Calculates visibility between satellites and ground stations.

Parameters:
- `satellite_positions`: Array of satellite positions (ECEF)
- `ground_stations`: Array of ground station coordinates (lat, lon, alt)
- `min_elevation`: Minimum elevation angle for visibility (degrees)

Returns a boolean array indicating visibility.

## Examples

TensorGator includes several example applications:

### Benchmark

Evaluates propagation performance with various constellation sizes and timesteps.

```bash
python -m tensorgator.examples.benchmark
```

### 3D Orbit Visualization

Interactive 3D visualization of satellite orbits.

```bash
python -m tensorgator.examples.3d_orbit_visualization
```

### Coverage Map

Generates global coverage maps for satellite constellations.

```bash
python -m tensorgator.examples.coverage_map
```

### Interactive Visibility

Interactive tool for analyzing satellite visibility from ground stations.

```bash
python -m tensorgator.examples.interactive_visibility
```

## Validation

TensorGator has been validated against:
- CPU-based propagator,Beyond (validated to <1m/day precision with float64 dtype)
- Poliastro (~several km/day discrepancies due to difference between integrated force model and tensorgator analytical model)

## Future Roadmap

- **Hardware Acceleration**: Support for acceleration libraries (TensorFlow, Jax) beyond CUDA
- **Additional Mission Simulation**: End-to-end satellite mission simulation capabilities
- **Higher-order Perturbation Models**: Add support for atmospheric drag, solar radiation pressure, and third-body gravity

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
