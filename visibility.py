import numpy as np
import math
from numba import njit, prange, cuda
from .constants import RE

@njit
def is_visible(sat_pos, ground_ecef, min_elevation):
    """Check if a satellite is visible from a ground station."""
    sat_vector = sat_pos - ground_ecef
    distance = np.linalg.norm(sat_vector)
    sat_unit_vector = sat_vector / distance
    zenith_unit_vector = ground_ecef / np.linalg.norm(ground_ecef)
    elevation = np.arcsin(np.dot(sat_unit_vector, zenith_unit_vector))
    return elevation >= min_elevation

@cuda.jit(device=True)
def is_visible_device(sat_pos, ground_ecef, min_elevation):
    """CUDA device function to check satellite visibility."""
    # Calculate satellite-to-ground vector
    sat_vector_x = sat_pos[0] - ground_ecef[0]
    sat_vector_y = sat_pos[1] - ground_ecef[1]
    sat_vector_z = sat_pos[2] - ground_ecef[2]
    
    # Calculate distance
    distance = math.sqrt(sat_vector_x*sat_vector_x + sat_vector_y*sat_vector_y + sat_vector_z*sat_vector_z)
    
    # Calculate unit vector
    sat_unit_x = sat_vector_x / distance
    sat_unit_y = sat_vector_y / distance
    sat_unit_z = sat_vector_z / distance
    
    # Calculate ground station magnitude
    ground_mag = math.sqrt(ground_ecef[0]*ground_ecef[0] + ground_ecef[1]*ground_ecef[1] + ground_ecef[2]*ground_ecef[2])
    
    # Calculate elevation angle
    dot_product = sat_unit_x*ground_ecef[0] + sat_unit_y*ground_ecef[1] + sat_unit_z*ground_ecef[2]
    elevation = math.asin(dot_product / ground_mag)
    
    return elevation >= min_elevation

@njit(parallel=True)
def batch_visibility_check(satellite_positions, ground_points, min_elevation):
    """
    Check visibility for multiple ground points in parallel using CPU.
    
    Args:
        satellite_positions: Array of shape (num_sats, num_times, 3)
        ground_points: Array of shape (num_points, 3)
        min_elevation: Minimum elevation angle in radians
        
    Returns:
        visibility: Boolean array of shape (num_points, num_times)
    """
    num_points = len(ground_points)
    num_sats = satellite_positions.shape[0]
    num_times = satellite_positions.shape[1]
    visibility = np.zeros((num_points, num_times), dtype=np.bool_)
    
    for p in prange(num_points):
        ground_pos = ground_points[p]
        for t in range(num_times):
            for s in range(num_sats):
                if is_visible(satellite_positions[s, t], ground_pos, min_elevation):
                    visibility[p, t] = True
                    break
    return visibility

@cuda.jit
def visibility_kernel(sat_positions, ground_points, min_elevation, visibility):
    """
    CUDA kernel for calculating visibility between satellites and ground points.
    
    Args:
        sat_positions: Array of shape (num_sats, num_times, 3)
        ground_points: Array of shape (num_points, 3)
        min_elevation: Minimum elevation angle in radians
        visibility: Output array of shape (num_points, num_times)
    """
    # Get 2D grid indices
    p_idx, t_idx = cuda.grid(2)
    
    # Check bounds
    if p_idx < ground_points.shape[0] and t_idx < sat_positions.shape[1]:
        # Check visibility against all satellites at this time
        for s in range(sat_positions.shape[0]):
            if is_visible_device(sat_positions[s, t_idx], ground_points[p_idx], min_elevation):
                visibility[p_idx, t_idx] = 1
                break

def calculate_visibility_cuda(satellite_positions, ground_points, min_elevation):
    """
    Calculate visibility for all ground points using CUDA.
    
    Args:
        satellite_positions: Array of shape (num_sats, num_times, 3)
        ground_points: Array of shape (num_points, 3)
        min_elevation: Minimum elevation angle in radians
        
    Returns:
        visibility: Boolean array of shape (num_points, num_times)
    """
    num_points = len(ground_points)
    num_times = satellite_positions.shape[1]
    
    # Prepare device arrays
    d_sat_positions = cuda.to_device(satellite_positions)
    d_ground_points = cuda.to_device(ground_points)
    d_visibility = cuda.to_device(np.zeros((num_points, num_times), dtype=np.int32))
    
    # Configure CUDA grid
    threads_per_block = (16, 16)
    blocks_per_grid_x = (num_points + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (num_times + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # Launch kernel
    visibility_kernel[blocks_per_grid, threads_per_block](
        d_sat_positions, d_ground_points, min_elevation, d_visibility
    )
    
    # Copy results back to host
    visibility = d_visibility.copy_to_host()
    
    return visibility.astype(bool)

def calculate_max_gaps(visibility, time_step):
    """
    Calculate maximum gap durations for each ground point.
    Non-JIT version that works reliably.
    
    Args:
        visibility: Boolean array of shape (num_points, num_times)
        time_step: Time step in seconds
        
    Returns:
        max_gaps: Array of shape (num_points,) with maximum gap duration in seconds
    """
    num_points = visibility.shape[0]
    num_times = visibility.shape[1]
    max_gaps = np.zeros(num_points)
    
    # Use tqdm if available for progress tracking
    try:
        from tqdm import tqdm
        iterator = tqdm(range(num_points), desc="Calculating gaps")
    except ImportError:
        iterator = range(num_points)
    
    for p in iterator:
        vis_timeline = visibility[p]
        
        # Find the longest sequence of False values (gaps)
        current_gap = 0
        max_gap = 0
        
        for t in range(num_times):
            if not vis_timeline[t]:  # Gap (not visible)
                current_gap += 1
            else:  # Visible
                max_gap = max(max_gap, current_gap)
                current_gap = 0
        
        # Check final gap
        max_gap = max(max_gap, current_gap)
        
        # Convert to time
        max_gaps[p] = max_gap * time_step
    
    return max_gaps

def process_in_chunks(satellite_elements, times, ground_points, min_elevation, chunk_size=1000):
    """
    Process visibility in time chunks to reduce memory usage.
    
    Args:
        satellite_elements: Array of shape (num_sats, 6)
        times: Array of shape (num_times,)
        ground_points: Array of shape (num_points, 3)
        min_elevation: Minimum elevation angle in radians
        chunk_size: Number of time steps to process in each chunk
        
    Returns:
        visibility: Boolean array of shape (num_points, num_times)
    """
    from .propagation import satellite_positions
    
    num_points = len(ground_points)
    num_times = len(times)
    visibility = np.zeros((num_points, num_times), dtype=bool)
    
    for t_start in range(0, num_times, chunk_size):
        t_end = min(t_start + chunk_size, num_times)
        times_chunk = times[t_start:t_end]
        
        # Propagate satellites for this time chunk
        positions_chunk = satellite_positions(times_chunk, satellite_elements, backend='cuda')
        
        # Calculate visibility for this chunk
        visibility_chunk = calculate_visibility_cuda(positions_chunk, ground_points, min_elevation)
        
        # Store results
        visibility[:, t_start:t_end] = visibility_chunk
    
    return visibility
