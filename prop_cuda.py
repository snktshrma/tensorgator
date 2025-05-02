import numpy as np
import math
from numba import cuda
from .constants import MU, J2, RE
from .coord_conv import batch_eci_to_ecef, calculate_gmst_from_seconds

def propagate_constellation_cuda(satellite_elements, times, return_frame='ecef', epochs=None, input_type='kepler'):
    mu = MU
    j2 = J2
    re = RE
    num_sats = len(satellite_elements)
    num_times = len(times)
    fpp = np.float32 #adjust to fp64 for sub meter accuracy

    if epochs is None:
        epochs = np.zeros(num_sats, dtype=fpp)
        times_seconds = times - times[0]
    else:
        times_seconds = times

    d_elements = cuda.to_device(np.asarray(satellite_elements, dtype=fpp))
    d_times = cuda.to_device(np.asarray(times_seconds, dtype=fpp))
    d_epochs = cuda.to_device(np.asarray(epochs, dtype=fpp))
    d_positions = cuda.device_array((num_sats, num_times, 3), dtype=fpp)

    threadsperblock = (16, 16)
    blockspergrid_x = (num_sats + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (num_times + threadsperblock[1] - 1) // threadsperblock[1]
    simulate_orbit_batch_cuda[(blockspergrid_x, blockspergrid_y), threadsperblock](
        mu, j2, re, d_elements, d_times, d_epochs, d_positions
    )

    positions_eci = d_positions.copy_to_host()

    if return_frame.lower() == 'ecef':
        positions_ecef = np.zeros_like(positions_eci)
        for t in range(num_times):
            gmst = calculate_gmst_from_seconds(times_seconds[t])
            positions_ecef[:, t, :] = batch_eci_to_ecef(positions_eci[:, t, :], gmst)
        return positions_ecef
    else:
        return positions_eci

@cuda.jit(fastmath=True)
def simulate_orbit_batch_cuda(mu, j2, re, elements, times, epochs, positions):
    sat_idx, time_idx = cuda.grid(2)
    num_sats = elements.shape[0]
    num_times = times.shape[0]

    if sat_idx < num_sats and time_idx < num_times:
        a = elements[sat_idx, 0]
        e = elements[sat_idx, 1]
        i = elements[sat_idx, 2]
        raan = elements[sat_idx, 3]
        argp = elements[sat_idx, 4]
        M0 = elements[sat_idx, 5]

        t = times[time_idx] - epochs[sat_idx]

        x, y, z = calculate_position_device(mu, a, e, i, M0, argp, raan, j2, re, t)

        positions[sat_idx, time_idx, 0] = x
        positions[sat_idx, time_idx, 1] = y
        positions[sat_idx, time_idx, 2] = z

@cuda.jit(device=True, fastmath=True)
def calculate_position_device(mu, a, e, i, M, w, raan, j2, re, t):
    n0 = math.sqrt(mu / a**3)
    j2_secular_scale = (n0 * re**2 * j2) / (a**2 * (1 - e**2)**2)

    d_raan_dt = -3/2 * j2_secular_scale * math.cos(i)
    d_argper_dt = 3/4 * j2_secular_scale * (4 - 5 * math.sin(i)**2)
    d_M_dt = 3/4 * j2_secular_scale * math.sqrt(1 - e**2) * (2 - 3 * math.sin(i)**2)

    raan_t = raan + d_raan_dt * t
    w_t = w + d_argper_dt * t
    M_t = M + (n0 + d_M_dt) * t

    E = kepler_equation_device(M_t, e)
    nu = 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2), math.sqrt(1 - e) * math.cos(E / 2))
    r = a * (1 - e * math.cos(E))

    x_orb = r * math.cos(nu)
    y_orb = r * math.sin(nu)

    cos_w = math.cos(w_t)
    sin_w = math.sin(w_t)
    cos_raan = math.cos(raan_t)
    sin_raan = math.sin(raan_t)
    cos_i = math.cos(i)
    sin_i = math.sin(i)

    x = (cos_raan * cos_w - sin_raan * sin_w * cos_i) * x_orb + (-cos_raan * sin_w - sin_raan * cos_w * cos_i) * y_orb
    y = (sin_raan * cos_w + cos_raan * sin_w * cos_i) * x_orb + (-sin_raan * sin_w + cos_raan * cos_w * cos_i) * y_orb
    z = (sin_w * sin_i) * x_orb + (cos_w * sin_i) * y_orb

    return x, y, z

@cuda.jit(device=True, fastmath=True)
def kepler_equation_device(M, e):
    tol = 1e-10
    max_iter = 10
    E = M
    for _ in range(max_iter):
        E_old = E
        f = E - e * math.sin(E) - M
        f_prime = 1 - e * math.cos(E)
        E = E - f / f_prime
        if abs(E - E_old) < tol:
            break
    return E
