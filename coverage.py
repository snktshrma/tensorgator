import numpy as np
from numba import njit
from .constants import RE

@njit
def is_visible(sat_pos, ground_ecef, min_elevation):
    sat_vector = sat_pos - ground_ecef
    distance = np.linalg.norm(sat_vector)
    sat_unit_vector = sat_vector / distance
    elevation = np.arcsin(np.dot(sat_unit_vector, ground_ecef) / np.linalg.norm(ground_ecef))
    return elevation >= min_elevation

def calculate_elevation(sat_pos, ground_ecef):
    sat_vector = sat_pos - ground_ecef
    distance = np.linalg.norm(sat_vector)
    sat_unit_vector = sat_vector / distance
    elevation = np.arcsin(np.dot(sat_unit_vector, ground_ecef) / np.linalg.norm(ground_ecef))
    return np.degrees(elevation)

def ground_station_ecef(lat, lon, radius=RE):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return np.array([x, y, z])
