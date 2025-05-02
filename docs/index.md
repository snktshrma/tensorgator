# TensorGator Documentation

Welcome to **TensorGator** — a CUDA-accelerated, massively parallel satellite orbit propagation package.

## Overview
TensorGator leverages GPU acceleration (CUDA) to simulate and propagate thousands of satellite orbits in parallel. Designed for high-performance space mission analysis, constellation studies, and research, it supports both ECI and ECEF coordinate systems with accurate sidereal time handling.

## Key Features
- **Massively Parallel Propagation:** Utilizes CUDA for large scale/long duration orbit calculations.
- **J2 Perturbation Model:** Accurate long-term propagation with Earth's oblateness effects.
- **Flexible Epoch Handling:** Supports satellite-specific epochs for advanced scenarios.
- **Batch Coordinate Conversion:** Fast ECI/ECEF transformations with Numba acceleration.
- **Python API:** Easy integration into Python workflows.

## Installation
```
pip install tensorgator
```
Or for development:
```
cd tensorgator
pip install -e .
```

## Quick Start Example
```python
import tensorgator as tg
import numpy as np

# Define initial positions and velocities (ECI, meters and m/s)
positions = np.array([[7000e3, 0, 0]])
velocities = np.array([[0, 7.5e3, 1.0e3]])

times = np.linspace(0, 3600, 100)  # 1 hour, 100 steps

# Propagate using CUDA
result = tg.propagate_cuda(positions, velocities, times)
print(result)  # Shape: (num_sats, num_times, 6)
```

## Documentation Contents
- [API Reference](api.md)
- [Coordinate Systems](coordinates.md)
- [Comparison Tests](comparison.md)
- [Examples](examples.md)

---

> For more details, see the API reference and example notebooks.
