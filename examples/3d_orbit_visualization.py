"""
3D Visualization of 1000 Satellites Orbiting Earth
==================================================

This example demonstrates:
1. Propagating 1000 satellites using tensorgator's CUDA backend
2. 3D visualization of Earth and satellite orbits
3. Animation of satellite movement over time
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import time
from tqdm import tqdm
import sys
import os

# Add the parent directory to the path to find tensorgator
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Now import tensorgator
import tensorgator as tg
from tensorgator.prop_cuda import propagate_constellation_cuda
def create_earth_sphere(ax, radius=1.0, resolution=50):
    """Create a 3D sphere representing Earth."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot the surface
    earth = ax.plot_surface(x, y, z, color='blue', alpha=0.3, 
                           linewidth=0, antialiased=True)
    
    # Add wireframe for better visibility
    wireframe = ax.plot_wireframe(x, y, z, color='blue', alpha=0.1, 
                                 linewidth=0.2, antialiased=True)
    
    return earth, wireframe

def main():
    print("Generating 10000 random satellites in 3D space...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Constants
    RE = tg.RE  # Earth radius in meters
    
    # Number of satellites
    num_sats = 4000
    
    # Generate random satellites
    constellation = []
    
    # Create different orbit categories
    orbit_types = {
        'LEO': {'count': 1000, 'alt_range': (300000, 2000000), 'inc_range': (20, 98)},
        'MEO': {'count': 1000, 'alt_range': (5000000, 20000000), 'inc_range': (0, 90)},
        'GEO': {'count': 1000, 'alt_range': (35786000, 35786000), 'inc_range': (0, 5)},
        'HEO': {'count': 1000, 'alt_range': (500000, 40000000), 'inc_range': (60, 90), 'ecc_range': (0.2, 0.7)}
    }
    
    # Track satellite categories for coloring
    sat_categories = []
    
    # Generate satellites for each orbit type
    for orbit_type, params in orbit_types.items():
        count = params['count']
        for _ in range(count):
            # Random altitude within range
            alt_min, alt_max = params['alt_range']
            altitude = np.random.uniform(alt_min, alt_max)
            a = RE + altitude
            
            # Eccentricity (circular by default, except for HEO)
            if orbit_type == 'HEO':
                e_min, e_max = params['ecc_range']
                e = np.random.uniform(e_min, e_max)
            else:
                e = 0.0
            
            # Random inclination within range
            inc_min, inc_deg_max = params['inc_range']
            inc = np.radians(np.random.uniform(inc_min, inc_deg_max))
            
            # Random RAAN, argument of perigee, and mean anomaly
            raan = np.radians(np.random.uniform(0, 360))
            argp = np.radians(np.random.uniform(0, 360))
            M0 = np.radians(np.random.uniform(0, 360))
            
            constellation.append([a, e, inc, raan, argp, M0])
            sat_categories.append(orbit_type)
    
    constellation = np.array(constellation)
    
    # Create a time span (1 hours with 5-second steps)
    sim_duration = 3600  # 1 hours in seconds
    time_step = 5  # 5 seconds
    times = np.arange(0, sim_duration, time_step)
    num_frames = len(times)
    
    print(f"Propagating {num_sats} satellites over {num_frames} time steps...")
    start_time = time.time()
    
    # Use CUDA propagation for better performance
    positions = propagate_constellation_cuda(constellation, times, return_frame='ecef')
    
    prop_time = time.time() - start_time
    print(f"Propagation completed in {prop_time:.2f} seconds")
    
    # Prepare for visualization
    print("Creating 3D visualization...")
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Add Earth
    earth_radius_scaled = 1.0  # Scaled Earth radius for visualization
    earth, wireframe = create_earth_sphere(ax, radius=earth_radius_scaled)
    
    # Scale factor to convert from meters to visualization units
    scale_factor = earth_radius_scaled / RE
    
    # Prepare satellite scatter plot
    # Define colors for different orbit types
    color_map = {
        'LEO': 'red',
        'MEO': 'green',
        'GEO': 'yellow',
        'HEO': 'magenta'
    }
    
    # Convert categories to colors
    colors = [color_map[cat] for cat in sat_categories]
    
    # Initial satellite positions (first frame)
    x_init = positions[:, 0, 0] * scale_factor
    y_init = positions[:, 0, 1] * scale_factor
    z_init = positions[:, 0, 2] * scale_factor
    
    # Initial scatter plot
    scatter = ax.scatter(x_init, y_init, z_init, s=2, alpha=0.8, c=colors)
    
    # Set axis limits
    max_alt = np.max(constellation[:, 0]) * scale_factor
    ax.set_xlim(-max_alt, max_alt)
    ax.set_ylim(-max_alt, max_alt)
    ax.set_zlim(-max_alt, max_alt)
    
    # Set labels
    ax.set_xlabel('X (Earth radii)')
    ax.set_ylabel('Y (Earth radii)')
    ax.set_zlabel('Z (Earth radii)')
    ax.set_title(f'1000 Satellites Orbiting Earth')
    
    # Add a legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label=f'LEO ({orbit_types["LEO"]["count"]})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label=f'MEO ({orbit_types["MEO"]["count"]})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=8, label=f'GEO ({orbit_types["GEO"]["count"]})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='magenta', markersize=8, label=f'HEO ({orbit_types["HEO"]["count"]})')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add time display
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
    
    # Animation update function
    def update(frame):
        # Scale positions for visualization
        x = positions[:, frame, 0] * scale_factor
        y = positions[:, frame, 1] * scale_factor
        z = positions[:, frame, 2] * scale_factor
        
        # Update scatter plot
        scatter._offsets3d = (x, y, z)
        
        # Update time display
        elapsed_minutes = frame * time_step / 60
        time_text.set_text(f'Time: {elapsed_minutes:.0f} minutes')
        
        # Fixed view (no rotation)
        ax.view_init(elev=25, azim=45)
        
        return scatter, time_text
    
    # Create animation
    print("Creating animation...")
    anim = FuncAnimation(fig, update, frames=range(0, num_frames, 5), 
                         interval=50, blit=False)
    
    # Save animation
    print("Saving animation (this may take a while)...")
    anim.save('./satellites_3d.gif', writer='pillow', fps=15, dpi=100)
    print("Animation saved as 'satellites_3d.gif'")
    
    # Show the visualization
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
