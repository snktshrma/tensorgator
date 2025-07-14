"""
Compare TensorGator's CUDA implementation to Poliastro's J2 propagator.
WE DO NOT EXPECT EXACT MATCHES DUE TO ANALYTIC VS INTEGRATED APPROACHES.

"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

from astropy import units as u
from astropy.time import Time, TimeDelta

from poliastro.twobody import orbit
from poliastro import constants
from poliastro.bodies import Earth
from poliastro.core.perturbations import J2_perturbation
from poliastro.core.elements import rv2coe
from poliastro.core.propagation import func_twobody
from poliastro.ephem import EpochsArray
from poliastro.twobody.propagation import CowellPropagator

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensorgator.propagation import satellite_positions

class TestPoliastroComparison(unittest.TestCase):
    """Test to compare TensorGator propagator with Poliastro."""
    
    def setUp(self):
        """Set up test parameters."""
        # Earth parameters
        self.mu = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)
        self.re = 6378137.0  # Earth's radius (m)
        self.j2 = 1.08262668e-3  # Earth's J2 coefficient
        
        # Test satellites in different orbit regimes
        # Format: [name, a (m), e, i (rad), RAAN (rad), AoP (rad), M (rad)]
        self.test_satellites = [
            # LEO satellite (similar to ISS)
            ["LEO-1", 6878000, 0.0003, np.radians(51.6), np.radians(45), np.radians(30), np.radians(0)],
            
            # MEO satellite (similar to GPS)
            ["MEO-1", 26560000, 0.0001, np.radians(55), np.radians(120), np.radians(60), np.radians(0)],
            
            # GEO satellite
            ["GEO-1", 42164000, 0.0001, np.radians(0.1), np.radians(75), np.radians(0), np.radians(180)],
            
            # Highly eccentric orbit (similar to Molniya)
            ["HEO-1", 26600000, 0.74, np.radians(63.4), np.radians(270), np.radians(270), np.radians(180)]
        ]
        
        # Convert to numpy array for TensorGator (excluding name)
        self.tg_elements = np.array([sat[1:] for sat in self.test_satellites])
        
        # Simulation time parameters
        self.epoch = '2000-01-01 00:00:00.0'
        self.epoch_time = Time(self.epoch, format='iso', scale='utc')
        self.sim_length = 2 * u.hour  # 2 hours simulation
        self.sim_resolution = 1 * u.min  # 1 minute resolution
        self.num_times = int((self.sim_length / self.sim_resolution).decompose().value) + 1
        # For HyperGator: epochs in seconds since J2000 for each satellite
        self.tg_epochs = np.zeros(len(self.test_satellites))  # All start at J2000
        self.prop_times = TimeDelta(np.linspace(0, self.sim_length.to(u.s).value, num=self.num_times) * u.s)
        
        # Time array for TensorGator (seconds)
        self.times = np.linspace(0, self.sim_length.to(u.s).value, self.num_times)
    
    def test_poliastro_comparison(self):
        """Test to compare TensorGator CUDA propagator with Poliastro."""
        print("\nTensorGator vs Poliastro Comparison Results:")
        
        # Propagate using TensorGator
        tg_positions = satellite_positions(
            self.times,
            self.tg_elements,
            backend='cuda',
            return_frame='eci',
            epochs=self.tg_epochs
        )
        
        # Propagate using Poliastro
        poliastro_positions = np.zeros((len(self.test_satellites), self.num_times, 3))
        from astropy import units as u

        for i, sat in enumerate(self.test_satellites):
            # Extract orbital elements
            name = sat[0]
            a = sat[1] * u.m  # Semi-major axis in meters
            e = float(sat[2]) * u.one  # Eccentricity (dimensionless)
            inc = float(sat[3]) * u.rad  # Inclination in radians
            raan = float(sat[4]) * u.rad  # RAAN in radians
            argp = float(sat[5]) * u.rad  # Argument of perigee in radians
            M0 = float(sat[6]) * u.rad  # Mean anomaly in radians
            
            # Create Poliastro orbit
            sat_orbit = orbit.Orbit.from_classical(
                Earth, 
                a, 
                e, 
                inc, 
                raan,
                argp, 
                M0,  
                epoch=self.epoch_time
            )
            
            def f(t0, state, k):
                du_kep = func_twobody(t0, state, k)
                ax, ay, az = J2_perturbation(
                    t0, state, k, J2=Earth.J2.value, R=Earth.R.to(u.km).value
                )
                du_ad = np.array([0, 0, 0, ax, ay, az])
                return du_kep + du_ad
            
            tofs = TimeDelta(np.linspace(0, self.sim_length.to(u.s).value, num=self.num_times) * u.s)
            
            ephem = sat_orbit.to_ephem(
                EpochsArray(sat_orbit.epoch + tofs, method=CowellPropagator(f=f))
            )
            
            positions_km = np.zeros((self.num_times, 3))
            
            for j, t in enumerate(tofs):
                r, _ = sat_orbit.propagate(t, method=CowellPropagator(f=f)).rv()
                
                positions_km[j, 0] = r[0].to(u.km).value
                positions_km[j, 1] = r[1].to(u.km).value
                positions_km[j, 2] = r[2].to(u.km).value
            
            poliastro_positions[i] = positions_km * 1000
        
        differences = np.zeros((len(self.test_satellites), self.num_times))
        percent_differences = np.zeros((len(self.test_satellites), self.num_times))
        
        max_diff = 0
        max_diff_percent = 0
        max_diff_sat = ""
        max_diff_time = 0
        
        for i, sat_data in enumerate(self.test_satellites):
            for t in range(self.num_times):
                diff = np.linalg.norm(tg_positions[i, t] - poliastro_positions[i, t])
                differences[i, t] = diff
                
                radius = np.linalg.norm(poliastro_positions[i, t])
                percent_differences[i, t] = (diff / radius) * 100
                
                if diff > max_diff:
                    max_diff = diff
                    max_diff_percent = percent_differences[i, t]
                    max_diff_sat = sat_data[0]
                    max_diff_time = self.times[t]
        
        print(f"Maximum position difference: {max_diff:.2f} meters")
        print(f"Maximum percentage difference: {max_diff_percent:.6f}%")
        print(f"Occurred for satellite {max_diff_sat} at time {max_diff_time} seconds")
        
        leo_idx = 0
        leo_diff = differences[leo_idx]
        
        final_diff = leo_diff[-1]
        
        sim_duration_days = self.sim_length.to(u.day).value
        diff_per_day = final_diff * (1.0 / sim_duration_days)
        
        print("\nLEO Satellite Analysis:")
        print(f"Simulation duration: {sim_duration_days:.2f} days")
        print(f"Final position difference: {final_diff:.2f} meters")
        print(f"Position difference per day: {diff_per_day:.2f} meters/day")
        print(f"Within expected tolerance (<2km/day): {diff_per_day < 2000}")
        
        intervals = 5
        interval_indices = np.linspace(0, len(self.times)-1, intervals, dtype=int)
        print("\nLEO position differences at intervals:")
        print("Time (hours) | Difference (meters)")
        print("-" * 35)
        for idx in interval_indices:
            time_hours = self.times[idx] / 3600.0
            print(f"{time_hours:10.2f} | {leo_diff[idx]:10.2f}")
        
        if leo_diff[-1] > leo_diff[0]:
            print("\nDivergence is increasing over time")
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(self.times / 86400.0, leo_diff)  # time in days
            print(f"Rate of divergence: {slope:.2f} meters/day (R² = {r_value**2:.4f})")
        else:
            print("\nDivergence is not increasing over time")
        
        plt.figure(figsize=(15, 12))
        
        plt.subplot(3, 1, 1)
        for i, sat_data in enumerate(self.test_satellites):
            plt.plot(self.times / 60, differences[i], label=sat_data[0], linewidth=2)
        
        plt.xlabel('Time (minutes)')
        plt.ylabel('Position Difference (meters)')
        plt.title('Absolute Position Differences Between HyperGator and Poliastro')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(3, 1, 2)
        for i, sat_data in enumerate(self.test_satellites):
            plt.plot(self.times / 60, percent_differences[i], label=sat_data[0], linewidth=2)
        
        plt.xlabel('Time (minutes)')
        plt.ylabel('Position Difference (%)')
        plt.title('Percentage Position Differences Relative to Orbit Radius')
        plt.grid(True)
        plt.legend()
        
        ax = plt.subplot(3, 1, 3, projection='3d')
        
        heo_idx = 3  # Index of the HEO satellite
        
        heo_tg = tg_positions[heo_idx]
        heo_poliastro = poliastro_positions[heo_idx]
        
        divergence = heo_tg - heo_poliastro
        
        scale_factor = 1.0
        if np.max(np.abs(divergence)) > 0:
            scale_factor = 1000.0 / np.max(np.abs(divergence))
        
        ax.plot(heo_poliastro[:, 0] / 1000, heo_poliastro[:, 1] / 1000, heo_poliastro[:, 2] / 1000, 'b-', label='Poliastro Path', linewidth=2)
        
        step = max(1, self.num_times // 10)  # Show 10 vectors along the path
        for t in range(0, self.num_times, step):
            ax.quiver(heo_poliastro[t, 0] / 1000, heo_poliastro[t, 1] / 1000, heo_poliastro[t, 2] / 1000,
                     divergence[t, 0] * scale_factor / 1000, divergence[t, 1] * scale_factor / 1000, divergence[t, 2] * scale_factor / 1000,
                     color='r', arrow_length_ratio=0.1)
        
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = (self.re / 1000) * np.cos(u) * np.sin(v)
        y = (self.re / 1000) * np.sin(u) * np.sin(v)
        z = (self.re / 1000) * np.cos(v)
        ax.plot_surface(x, y, z, color='g', alpha=0.1)
        
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title(f'Position Divergence for {self.test_satellites[heo_idx][0]} (Scaled by {scale_factor:.1f})')
        
        max_range = np.array([heo_poliastro[:, 0].max() - heo_poliastro[:, 0].min(),
                             heo_poliastro[:, 1].max() - heo_poliastro[:, 1].min(),
                             heo_poliastro[:, 2].max() - heo_poliastro[:, 2].min()]).max() / 2.0
        mid_x = (heo_poliastro[:, 0].max() + heo_poliastro[:, 0].min()) * 0.5 / 1000
        mid_y = (heo_poliastro[:, 1].max() + heo_poliastro[:, 1].min()) * 0.5 / 1000
        mid_z = (heo_poliastro[:, 2].max() + heo_poliastro[:, 2].min()) * 0.5 / 1000
        ax.set_xlim(mid_x - max_range / 1000, mid_x + max_range / 1000)
        ax.set_ylim(mid_y - max_range / 1000, mid_y + max_range / 1000)
        ax.set_zlim(mid_z - max_range / 1000, mid_z + max_range / 1000)
        
        plt.tight_layout()
        plt.savefig('poliastro_comparison.png', dpi=300)
        print("Enhanced visualization saved as 'poliastro_comparison.png'")
        
        plt.figure(figsize=(15, 10))
        
        ax = plt.subplot(111, projection='3d')
        
        heo_tg_pos = tg_positions[heo_idx]
        heo_poliastro_pos = poliastro_positions[heo_idx]
        
        ax.plot(heo_poliastro_pos[:, 0] / 1000, heo_poliastro_pos[:, 1] / 1000, heo_poliastro_pos[:, 2] / 1000, 'b-', linewidth=2, label='Poliastro')
        ax.plot(heo_tg_pos[:, 0] / 1000, heo_tg_pos[:, 1] / 1000, heo_tg_pos[:, 2] / 1000, 'r-', linewidth=2, label='HyperGator')
        
        time_markers = np.linspace(0, self.num_times-1, 5).astype(int)
        for t in time_markers:
            ax.scatter(heo_poliastro_pos[t, 0] / 1000, heo_poliastro_pos[t, 1] / 1000, heo_poliastro_pos[t, 2] / 1000, 
                      c='b', s=50, marker='o')
            ax.scatter(heo_tg_pos[t, 0] / 1000, heo_tg_pos[t, 1] / 1000, heo_tg_pos[t, 2] / 1000, 
                      c='r', s=50, marker='^')
            if t == time_markers[0]:
                ax.text(heo_poliastro_pos[t, 0] / 1000, heo_poliastro_pos[t, 1] / 1000, heo_poliastro_pos[t, 2] / 1000, 
                       f'{self.times[t]/60:.0f} min', color='k')
        
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = (self.re / 1000) * np.cos(u) * np.sin(v)
        y = (self.re / 1000) * np.sin(u) * np.sin(v)
        z = (self.re / 1000) * np.cos(v)
        ax.plot_surface(x, y, z, color='g', alpha=0.2)
        
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title(f'Highly Eccentric Orbit Comparison: {self.test_satellites[heo_idx][0]}')
        ax.legend()
        
        max_range = np.array([
            max(heo_poliastro_pos[:, 0].max(), heo_tg_pos[:, 0].max()) - min(heo_poliastro_pos[:, 0].min(), heo_tg_pos[:, 0].min()),
            max(heo_poliastro_pos[:, 1].max(), heo_tg_pos[:, 1].max()) - min(heo_poliastro_pos[:, 1].min(), heo_tg_pos[:, 1].min()),
            max(heo_poliastro_pos[:, 2].max(), heo_tg_pos[:, 2].max()) - min(heo_poliastro_pos[:, 2].min(), heo_tg_pos[:, 2].min())
        ]).max() / 2.0
        
        mid_x = (max(heo_poliastro_pos[:, 0].max(), heo_tg_pos[:, 0].max()) + min(heo_poliastro_pos[:, 0].min(), heo_tg_pos[:, 0].min())) * 0.5 / 1000
        mid_y = (max(heo_poliastro_pos[:, 1].max(), heo_tg_pos[:, 1].max()) + min(heo_poliastro_pos[:, 1].min(), heo_tg_pos[:, 1].min())) * 0.5 / 1000
        mid_z = (max(heo_poliastro_pos[:, 2].max(), heo_tg_pos[:, 2].max()) + min(heo_poliastro_pos[:, 2].min(), heo_tg_pos[:, 2].min())) * 0.5 / 1000
        
        ax.set_xlim(mid_x - max_range / 1000, mid_x + max_range / 1000)
        ax.set_ylim(mid_y - max_range / 1000, mid_y + max_range / 1000)
        ax.set_zlim(mid_z - max_range / 1000, mid_z + max_range / 1000)
        
        plt.tight_layout()
        plt.savefig('heo_orbit_comparison.png', dpi=300)
        print("HEO orbit comparison saved as 'heo_orbit_comparison.png'")
        
        plt.figure(figsize=(15, 10))
        
        for i, sat_data in enumerate(self.test_satellites):
            plt.subplot(2, 2, i+1)
            
            sat_name = sat_data[0]
            
            for j, comp in enumerate(['X', 'Y', 'Z']):
                diff_comp = tg_positions[i, :, j] - poliastro_positions[i, :, j]
                plt.plot(self.times / 60, diff_comp, label=f'{comp} Component', linewidth=2)
            
            plt.xlabel('Time (minutes)')
            plt.ylabel('Position Difference (meters)')
            plt.title(f'Position Component Differences for {sat_name}')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('poliastro_component_differences.png', dpi=300)
        print("Component differences visualization saved as 'poliastro_component_differences.png'")
        
        results_data = {
            'Time (s)': self.times
        }
        
        for i, sat_data in enumerate(self.test_satellites):
            sat_name = sat_data[0]
            results_data[f'{sat_name}_Diff_X'] = tg_positions[i, :, 0] - poliastro_positions[i, :, 0]
            results_data[f'{sat_name}_Diff_Y'] = tg_positions[i, :, 1] - poliastro_positions[i, :, 1]
            results_data[f'{sat_name}_Diff_Z'] = tg_positions[i, :, 2] - poliastro_positions[i, :, 2]
            results_data[f'{sat_name}_Diff_Total'] = differences[i]
            results_data[f'{sat_name}_Diff_Percent'] = percent_differences[i]
        
        df = pd.DataFrame(results_data)
        df.to_excel('tensorgator_poliastro_comparison.xlsx', index=False)
        print("Detailed comparison data saved to 'tensorgator_poliastro_comparison.xlsx'")

if __name__ == '__main__':
    unittest.main()
