"""
Compare TensorGator's CPU and CUDA implementations to ensure they produce identical results.

"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for TensorGator import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorgator.propagation import satellite_positions
from tensorgator.constants import MU, RE, J2

def main():
    # --- ISS-like orbit parameters---
    a = RE + 400e3
    e = 0.0
    i = np.radians(51.6)
    raan = 0.0
    argp = 0.0
    M = 0.0

    dt = 300  # 5 minutes
    n_steps = 288  # 24 hours
    times = np.arange(0, dt*n_steps, dt)

    print("TensorGator constants:")
    print(f"  Earth radius (RE): {RE}")
    print(f"  Earth J2: {J2}")
    print(f"  Earth GM (MU): {MU}")
    print("")

    # --- TensorGator CPU Propagation ---
    tg_elements = np.array([[a, e, i, raan, argp, M]])
    tg_cpu_positions = satellite_positions(times, tg_elements, backend='cpu', return_frame='eci')
    tg_cpu_positions = tg_cpu_positions[0]

    # --- TensorGator CUDA Propagation ---
    tg_cuda_positions = satellite_positions(times, tg_elements, backend='cuda', return_frame='eci')
    tg_cuda_positions = tg_cuda_positions[0]

    diff = np.linalg.norm(tg_cpu_positions - tg_cuda_positions, axis=1)
    print(f"Max difference: {np.max(diff):.6f} m, Mean difference: {np.mean(diff):.6f} m")

    import pandas as pd
    df = pd.DataFrame({
        'time_s': times,
        'cpu_x': tg_cpu_positions[:,0],
        'cpu_y': tg_cpu_positions[:,1],
        'cpu_z': tg_cpu_positions[:,2],
        'cuda_x': tg_cuda_positions[:,0],
        'cuda_y': tg_cuda_positions[:,1],
        'cuda_z': tg_cuda_positions[:,2],
        'diff_m': diff
    })
    df.to_csv('cpu_cuda_comparison_intermediates.csv', index=False)

    max_idx = np.argmax(diff)
    print("\nFirst step:")
    print(df.iloc[0])
    print("\nMax difference step:")
    print(df.iloc[max_idx])
    print("\nLast step:")
    print(df.iloc[-1])

    plt.figure(figsize=(8,4))
    plt.plot(times/3600, diff)
    plt.xlabel("Time [hours]")
    plt.ylabel("Position difference [m]")
    plt.title("CPU vs CUDA J2 Propagator Position Difference\n ~1m/day Due to Float32 vs Float64 Precision")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cpu_cuda_comparison.png')
    plt.show()

if __name__ == "__main__":
    main()
