import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from mpl_toolkits.basemap import Basemap
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to find tensorgator
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import TensorGator modules
import tensorgator as tg
from tensorgator.prop_cuda import propagate_constellation_cuda
from tensorgator.coord_conv import batch_eci_to_ecef, calculate_gmst, cart_to_lat_lon
from tensorgator.constants import RE
from tensorgator.visibility import is_visible

def ground_station_ecef(lat_deg, lon_deg, radius):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return np.array([x, y, z])

def create_interactive_visualization():
    """Create an interactive visualization with time slider and visibility links."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Constants
    RE = tg.RE  # Earth radius in meters
    
    # Generate 5 random satellites
    num_sats = 5
    constellation = []
    
    # Create different orbit categories
    orbit_types = {
        'LEO': {'count': 3, 'alt_range': (500000, 1000000), 'inc_range': (0, 90)},
        'GEO': {
            'count': 2, 
            # Exact GEO altitude: 35,786 km above Earth's surface
            'alt_range': (35786000, 35786000),  
            'inc_range': (0, 0)
        }
    }
    
    # Generate satellites for each orbit type
    sat_names = []
    for orbit_type, params in orbit_types.items():
        count = params['count']
        for i in range(count):
            # Random altitude within range
            alt_min, alt_max = params['alt_range']
            altitude = np.random.uniform(alt_min, alt_max)
            a = RE + altitude
            
            # Circular orbit
            e = 0.0
            
            # Random inclination within range
            inc_min, inc_deg_max = params['inc_range']
            inc = np.radians(np.random.uniform(inc_min, inc_deg_max))
            
            # For GEO satellites, ensure they are truly geostationary
            if orbit_type == 'GEO':
                # Place at specific longitudes (spaced around the equator)
                longitude = (i * 180.0) % 360  # Space them 180 degrees apart
                raan = np.radians(longitude)
                argp = 0.0  # No argument of perigee for circular orbit
                M0 = np.radians(0.0)  # Position satellite at the chosen longitude
                
                # Ensure exact GEO parameters
                a = 42164000  # Exact GEO radius in meters (from Earth center)
                e = 0.0  # Perfectly circular
                inc = 0.0  # Zero inclination
            else:
                # Random RAAN, argument of perigee, and mean anomaly for non-GEO
                raan = np.radians(np.random.uniform(0, 360))
                argp = np.radians(np.random.uniform(0, 360))
                M0 = np.radians(np.random.uniform(0, 360))
            
            constellation.append([a, e, inc, raan, argp, M0])
            sat_names.append(f"{orbit_type}-{i+1}")
    
    constellation = np.array(constellation)
    
    # Create ground stations
    ground_stations = [
        {"name": "New York", "lat": 40.7, "lon": -74.0},
        {"name": "Tokyo", "lat": 35.7, "lon": 139.7},
        {"name": "London", "lat": 51.5, "lon": -0.1},
        {"name": "Sydney", "lat": -33.9, "lon": 151.2},
        {"name": "Rio de Janeiro", "lat": -22.9, "lon": -43.2}
    ]
    
    # Convert ground stations to ECEF
    ground_points = []
    for station in ground_stations:
        ground_points.append(ground_station_ecef(station["lat"], station["lon"], RE/1000))
    ground_points = np.array(ground_points)
    
    # Create a time span (2 hours with 1-minute steps)
    sim_duration = 2 * 3600  # 2 hours in seconds
    time_step = 60  # 1 minute
    times = np.arange(0, sim_duration, time_step)
    num_frames = len(times)
    
    # Start epoch for ECI/ECEF conversions
    epoch_start = datetime.now()
    
    # Create satellite epochs (all same for this example)
    j2000 = datetime(2000, 1, 1, 0, 0, 0)
    current_epoch_seconds = (epoch_start - j2000).total_seconds()
    epochs = np.ones(len(constellation)) * current_epoch_seconds
    
    
    # Propagate satellites in ECI frame
    print(f"Propagating {num_sats} satellites over {num_frames} time steps...")
    positions_eci = propagate_constellation_cuda(constellation, times, return_frame='ec', epochs=epochs)
    
    # Convert to ECEF manually at each time step
    positions_ecef = np.zeros_like(positions_eci)
    for t in range(num_frames):
        # Calculate GMST for this time - Convert numpy.int32 to Python int
        current_epoch = epoch_start + timedelta(seconds=float(times[t]))  # Convert to float first
        gmst = calculate_gmst(current_epoch)
        
        # Convert all satellite positions for this time step
        positions_ecef[:, t] = batch_eci_to_ecef(positions_eci[:, t], gmst)
    
    # Calculate visibility for all ground stations at all times
    min_elevation_deg = 10
    min_elevation_rad = np.radians(min_elevation_deg)
    
    # Initialize visibility matrix (ground_stations × satellites × times)
    visibility = np.zeros((len(ground_stations), num_sats, num_frames), dtype=bool)
    
    # Calculate visibility
    print("Calculating visibility...")
    for t in range(num_frames):
        for g, ground_pos in enumerate(ground_points):
            for s in range(num_sats):
                # Check visibility using optimized Numba function
                if is_visible(positions_ecef[s, t], ground_pos, min_elevation_rad):
                    visibility[g, s, t] = True
    
    # Check GEO satellite positions in ECEF frame
    print("Checking GEO satellite positions in ECEF frame...")
    for s in range(num_sats):
        if "GEO" in sat_names[s]:
            # Get positions at first and last time steps
            first_pos = positions_ecef[s, 0]
            last_pos = positions_ecef[s, -1]
            
            # Calculate displacement
            displacement = np.linalg.norm(last_pos - first_pos)
            
            # Convert to lat/lon for both positions
            first_lat, first_lon = cart_to_lat_lon(*first_pos)
            last_lat, last_lon = cart_to_lat_lon(*last_pos)
            
            print(f"{sat_names[s]}: Displacement = {displacement/1000:.2f} km")
            print(f"  Starting position: Lat={first_lat:.2f}°, Lon={first_lon:.2f}°")
            print(f"  Ending position: Lat={last_lat:.2f}°, Lon={last_lon:.2f}°")
            
            # GEO satellites should remain at nearly the same position in ECEF
            if displacement > 100000:  # More than 100 km is suspicious for GEO
                print(f"  WARNING: {sat_names[s]} shows significant movement in ECEF frame!")
    
    # Create the visualization
    print("Creating interactive visualization...")
    fig = plt.figure(figsize=(12, 8))
    
    # Create a Basemap instance
    m = Basemap(projection='mill', lon_0=0, resolution='c')
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
    m.drawparallels(np.arange(-90, 91, 30), labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(-180, 181, 60), labels=[0, 0, 0, 1])
    
    # Plot ground stations
    ground_x, ground_y = [], []
    for station in ground_stations:
        x, y = m(station["lon"], station["lat"])
        ground_x.append(x)
        ground_y.append(y)
    
    # Plot ground stations
    ground_scatter = plt.scatter(ground_x, ground_y, s=50, c='blue', marker='^', label='Ground Stations')
    
    # Add ground station labels
    for i, station in enumerate(ground_stations):
        x, y = m(station["lon"], station["lat"])
        plt.text(x + 100000, y + 100000, station["name"], fontsize=8)
    
    # Initialize satellite scatter plot
    sat_scatter = plt.scatter([], [], s=50, c='red', marker='o', label='Satellites')
    
    # Initialize visibility lines
    vis_lines = []
    for g in range(len(ground_stations)):
        for s in range(num_sats):
            line, = plt.plot([], [], 'g-', alpha=0.5, linewidth=1)
            vis_lines.append(line)
    
    # Add a legend
    plt.legend(loc='upper right')
    
    # Add time display
    time_text = plt.text(0.02, 0.95, '', transform=plt.gca().transAxes, fontsize=10)
    
    # Add slider for time control
    ax_slider = plt.axes([0.25, 0.02, 0.65, 0.03])
    time_slider = Slider(ax_slider, 'Time (min)', 0, (num_frames-1)/60, valinit=0)
    
    # Function to update the plot based on slider value
    def update_plot(val):
        # Get the current frame index
        frame = int(val * 60)
        if frame >= num_frames:
            frame = num_frames - 1
            
        # Current epoch - Convert numpy.int32 to Python int/float
        current_epoch = epoch_start + timedelta(seconds=float(times[frame]))
        
        # Update time display
        time_text.set_text(f'Time: {current_epoch.strftime("%Y-%m-%d %H:%M:%S")}')
        
        # Update satellite positions
        sat_x, sat_y = [], []
        for s in range(num_sats):
            sat_pos_ecef = positions_ecef[s, frame]
            
            # Convert to lat/lon
            lat, lon = cart_to_lat_lon(*sat_pos_ecef)
            x, y = m(lon, lat)
            sat_x.append(x)
            sat_y.append(y)
        
        # Update satellite scatter plot
        sat_scatter.set_offsets(np.c_[sat_x, sat_y])
        
        # Update visibility lines
        line_idx = 0
        for g in range(len(ground_stations)):
            for s in range(num_sats):
                if visibility[g, s, frame]:
                    # Satellite is visible from this ground station
                    vis_lines[line_idx].set_data([ground_x[g], sat_x[s]], [ground_y[g], sat_y[s]])
                else:
                    # Not visible, clear the line
                    vis_lines[line_idx].set_data([], [])
                line_idx += 1
        
        # Redraw the figure
        fig.canvas.draw_idle()
    
    # Connect the slider to the update function
    time_slider.on_changed(update_plot)
    
    # Initialize the plot with the first frame
    update_plot(0)
    
    # Add title
    plt.title('Interactive Satellite Visualization with Visibility Links')
    
    # Show the plot
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for the slider
    plt.show()
    
    # Create animation
    print("Creating animation...")
    
    def init():
        sat_scatter.set_offsets(np.c_[[], []])
        for line in vis_lines:
            line.set_data([], [])
        return [sat_scatter] + vis_lines + [time_text]
    
    def animate(frame):
        # Current epoch - Convert numpy.int32 to Python int/float
        current_epoch = epoch_start + timedelta(seconds=float(times[frame * 5]))
        
        # Update time display
        time_text.set_text(f'Time: {current_epoch.strftime("%Y-%m-%d %H:%M:%S")}')
        
        # Update satellite positions
        sat_x, sat_y = [], []
        for s in range(num_sats):
            # Use ECEF positions directly
            sat_pos_ecef = positions_ecef[s, frame * 5]
            
            # Convert to lat/lon
            lat, lon = cart_to_lat_lon(*sat_pos_ecef)
            x, y = m(lon, lat)
            sat_x.append(x)
            sat_y.append(y)
        
        # Update satellite scatter plot
        sat_scatter.set_offsets(np.c_[sat_x, sat_y])
        
        # Update visibility lines
        line_idx = 0
        for g in range(len(ground_stations)):
            for s in range(num_sats):
                if visibility[g, s, frame * 5]:
                    # Satellite is visible from this ground station
                    vis_lines[line_idx].set_data([ground_x[g], sat_x[s]], [ground_y[g], sat_y[s]])
                else:
                    # Not visible, clear the line
                    vis_lines[line_idx].set_data([], [])
                line_idx += 1
        
        return [sat_scatter] + vis_lines + [time_text]
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=range(0, num_frames // 5),
                          init_func=init, interval=100, blit=True)
    
    # Save animation
    anim.save('./satellite_visibility.gif', writer='pillow', fps=10, dpi=100)
    print("Animation saved as 'satellite_visibility.gif'")

if __name__ == "__main__":
    create_interactive_visualization()