import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Add tensorgator to path (similar to your example)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import tensorgator as tg
from tensorgator.prop_cuda import propagate_constellation_cuda

def batch_propagate(constellation, times, batch_size_sats=None, batch_size_times=None, return_frame='ecef'):
    """
    Propagate a large constellation in batches to avoid GPU memory issues.
    
    Args:
        constellation: Array of satellite elements
        times: Array of times to propagate
        batch_size_sats: Maximum number of satellites per batch (None for no batching)
        batch_size_times: Maximum number of timesteps per batch (None for no batching)
        return_frame: Frame to return positions in ('ecef' or 'eci')
    
    Returns:
        positions: Array of satellite positions
    """
    num_sats = len(constellation)
    num_times = len(times)
    
    # Determine batch sizes
    if batch_size_sats is None:
        batch_size_sats = num_sats
    if batch_size_times is None:
        batch_size_times = num_times
    
    # Initialize result array
    positions = np.zeros((num_sats, num_times, 3), dtype=np.float32)
    
    # Process in batches
    for sat_start in range(0, num_sats, batch_size_sats):
        sat_end = min(sat_start + batch_size_sats, num_sats)
        sat_batch = constellation[sat_start:sat_end]
        
        for time_start in range(0, num_times, batch_size_times):
            time_end = min(time_start + batch_size_times, num_times)
            time_batch = times[time_start:time_end]
            
            # Propagate this batch
            batch_positions = propagate_constellation_cuda(
                sat_batch, time_batch, return_frame=return_frame)
            
            # Store in result array
            positions[sat_start:sat_end, time_start:time_end, :] = batch_positions
    
    return positions

def estimate_optimal_batch_size(max_sats, max_times):
    """
    Estimate optimal batch sizes based on problem size and available memory.
    This is a simple heuristic and may need adjustment based on GPU specs.
    
    Args:
        max_sats: Maximum number of satellites
        max_times: Maximum number of timesteps
    
    Returns:
        batch_size_sats, batch_size_times: Recommended batch sizes
    """
    # Estimate memory per satellite-timestep (bytes)
    # Each position is 3 float32 values (12 bytes) plus overhead
    bytes_per_sat_time = 20  # Approximate
    
    # Estimate available GPU memory (adjust based on your GPU)
    # This is conservative to leave room for other GPU operations
    available_memory = 10 * 1024 * 1024 * 1024  # 1 GB (conservative)
    
    # Calculate total memory needed
    total_memory = max_sats * max_times * bytes_per_sat_time
    
    if total_memory <= available_memory:
        # No batching needed
        return max_sats, max_times
    
    # Calculate batch sizes to fit in memory
    # Try to keep satellite batches larger when possible
    memory_ratio = available_memory / total_memory
    
    # Start with square root allocation and adjust
    batch_size_sats = int(np.sqrt(memory_ratio * max_sats * max_times))
    
    # Prefer processing more satellites at once when possible
    if batch_size_sats > max_sats / 2:
        batch_size_sats = max_sats
        batch_size_times = int(available_memory / (batch_size_sats * bytes_per_sat_time))
    else:
        # Balance between satellites and timesteps
        batch_size_times = int(available_memory / (batch_size_sats * bytes_per_sat_time))
        
    # Ensure minimum batch sizes
    batch_size_sats = max(100, min(batch_size_sats, max_sats))
    batch_size_times = max(100, min(batch_size_times, max_times))
    
    return batch_size_sats, batch_size_times

def benchmark_propagation(satellite_counts, timestep_counts, num_runs=5):
    """
    Benchmark propagation performance with various satellite counts and timesteps,
    averaging over multiple runs and using batching to avoid memory issues.
    
    Args:
        satellite_counts: List of satellite counts to test
        timestep_counts: List of timestep counts to test
        num_runs: Number of runs to average over
    
    Returns:
        results: Dictionary with benchmark results
    """
    results = {
        'satellite_counts': satellite_counts,
        'timestep_counts': timestep_counts,
        'execution_times': np.zeros((len(satellite_counts), len(timestep_counts))),
        'std_devs': np.zeros((len(satellite_counts), len(timestep_counts))),
        'batch_sizes': [],  # Track batch sizes used
    }
    
    # Earth radius
    RE = tg.RE
    
    # Generate random seed for reproducibility
    np.random.seed(42)
    
    # Run benchmarks
    for i, num_sats in enumerate(satellite_counts):
        print(f"Testing with {num_sats} satellites...")
        
        # Generate satellite constellation
        constellation = []
        
        # Create parameters for random satellites (simplified from original)
        alt_range = (300000, 2000000)  # LEO range
        inc_range = (20, 98)
        
        for _ in range(num_sats):
            # Random altitude within range
            altitude = np.random.uniform(*alt_range)
            a = RE + altitude
            
            # Circular orbit
            e = 0.0
            
            # Random inclination
            inc = np.radians(np.random.uniform(*inc_range))
            
            # Random RAAN, argument of perigee, and mean anomaly
            raan = np.radians(np.random.uniform(0, 360))
            argp = np.radians(np.random.uniform(0, 360))
            M0 = np.radians(np.random.uniform(0, 360))
            
            constellation.append([a, e, inc, raan, argp, M0])
        
        constellation = np.array(constellation)
        
        for j, num_timesteps in enumerate(timestep_counts):
            print(f"  Testing with {num_timesteps} timesteps...")
            
            # Create time array
            time_step = 5  # 5 seconds, same as original
            times = np.arange(0, num_timesteps * time_step, time_step)
            
            # Estimate optimal batch sizes for this configuration
            batch_size_sats, batch_size_times = estimate_optimal_batch_size(num_sats, num_timesteps)
            results['batch_sizes'].append((num_sats, num_timesteps, batch_size_sats, batch_size_times))
            
            print(f"  Using batch sizes: {batch_size_sats} satellites, {batch_size_times} timesteps")
            
            # Run multiple times and average
            run_times = []
            for run in range(num_runs):
                print(f"    Run {run+1}/{num_runs}...")
                
                # Measure propagation time
                if run==0:#warmup
                    batch_propagate(constellation, times, 
                               batch_size_sats=batch_size_sats, 
                               batch_size_times=batch_size_times)
                start_time = time.time()
                
                # Use batched propagation
                batch_propagate(constellation, times, 
                               batch_size_sats=batch_size_sats, 
                               batch_size_times=batch_size_times)
                
                elapsed_time = time.time() - start_time
                run_times.append(elapsed_time)
                print(f"    Completed in {elapsed_time:.2f} seconds")
            
            # Calculate average and standard deviation
            avg_time = np.mean(run_times)
            std_dev = np.std(run_times)
            
            results['execution_times'][i, j] = avg_time
            results['std_devs'][i, j] = std_dev
            print(f"  Average: {avg_time:.2f} seconds (±{std_dev:.2f})")
    
    return results

def plot_results(results):
    """Plot scaling relationship between satellites and time for different timesteps."""
    satellite_counts = results['satellite_counts']
    timestep_counts = results['timestep_counts']
    execution_times = results['execution_times']
    std_devs = results['std_devs']
    batch_sizes = results['batch_sizes']
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Define color map for different timestep counts
    colors = plt.cm.viridis(np.linspace(0, 1, len(timestep_counts)))
    
    # Plot lines for each timestep count
    for j, timestep in enumerate(timestep_counts):
        plt.errorbar(satellite_counts, execution_times[:, j], 
                    yerr=std_devs[:, j],
                    marker='o', linestyle='-', color=colors[j],
                    label=f'{timestep} timesteps', capsize=5)
    
    # Set labels and title
    plt.xlabel('Number of Satellites', fontsize=14)
    plt.ylabel('Execution Time (seconds)', fontsize=14)
    plt.title('Scaling Relationship: Number of Satellites vs. Propagation Time\n(Average of 5 runs with Batching)', fontsize=16)
    
    # Add trend line fit for each timestep count
    for j, timestep in enumerate(timestep_counts):
        # Use polynomial fit for visualization (can switch to other models if needed)
        z = np.polyfit(satellite_counts, execution_times[:, j], 2)
        p = np.poly1d(z)
        
        # Create smooth line for plotting
        x_smooth = np.linspace(min(satellite_counts), max(satellite_counts), 100)
        plt.plot(x_smooth, p(x_smooth), '--', color=colors[j], alpha=0.7)
    
    # Customize grid and legend
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(title="Propagation Length", fontsize=12, title_fontsize=12)
    
    # Set axis scales
    plt.xscale('log')
    plt.yscale('log')
    
    # Customize ticks
    plt.xticks(satellite_counts, [str(s) for s in satellite_counts])
    
    # Add annotations about scaling behavior
    # Calculate approximate scaling factor (if it follows power law O(n^k))
    for j, timestep in enumerate(timestep_counts):
        if len(satellite_counts) >= 3:
            # Use last few points to estimate scaling
            x1, x2 = satellite_counts[-2], satellite_counts[-1]
            y1, y2 = execution_times[-2, j], execution_times[-1, j]
            k = np.log(y2/y1) / np.log(x2/x1)
            
            # Add annotation
            plt.annotate(f'O(n^{k:.2f})',
                        xy=(satellite_counts[-1], execution_times[-1, j]),
                        xytext=(10, 0), textcoords='offset points',
                        color=colors[j], fontsize=10)
    
    # Add text for execution environment information
    info_text = "GPU: CUDA backend (tensorgator)\n"
    info_text += "With memory-optimized batching"
    plt.figtext(0.02, 0.02, info_text, fontsize=10, alpha=0.7)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('satellite_propagation_scaling_batched.png', dpi=300)
    
    # Show plot
    plt.show()
    
    # Print scaling analysis
    print("\nScaling Analysis:")
    print("=================")
    for j, timestep in enumerate(timestep_counts):
        if len(satellite_counts) >= 3:
            # Use last few points to estimate scaling
            x1, x2 = satellite_counts[-2], satellite_counts[-1]
            y1, y2 = execution_times[-2, j], execution_times[-1, j]
            k = np.log(y2/y1) / np.log(x2/x1)
            print(f"For {timestep} timesteps: Time complexity approximately O(n^{k:.2f})")
    
    # Print batch size information
    print("\nBatch Sizes Used:")
    print("=================")
    for num_sats, num_times, batch_sats, batch_times in batch_sizes:
        num_batches_sats = np.ceil(num_sats / batch_sats)
        num_batches_times = np.ceil(num_times / batch_times)
        total_batches = num_batches_sats * num_batches_times
        print(f"Config: {num_sats} satellites, {num_times} timesteps")
        print(f"  Batch sizes: {batch_sats} satellites, {batch_times} timesteps")
        print(f"  Total batches: {total_batches:.0f} ({num_batches_sats:.0f} × {num_batches_times:.0f})")

def main():
    # Define satellite counts to test (powers of 2 for better scaling analysis)
    satellite_counts = [1000, 5000, 10000, 20000, 40000, int(1e6)]
    
    # Define timestep counts to test
    timestep_counts = [100, 250, 500, 750, 1000]
    
    # Run benchmarks (with 5 runs for averaging)
    results = benchmark_propagation(satellite_counts, timestep_counts, num_runs=5)
    
    # Plot results
    plot_results(results)

if __name__ == "__main__":
    main()