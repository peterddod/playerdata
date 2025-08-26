import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def bad_acceleration_percentage(speed, time, max_accel=8, max_decel=-9):
    """
    Calculate the percentage of bad accelerations in the dataset.
    """
    # Calculate acceleration between consecutive time points
    accel = np.diff(speed) / np.diff(time)

    # Count how many accelerations exceed physical limits
    bad_accel = (accel > max_accel) | (accel < max_decel)  # m/s²
    bad_pct = 100 * np.sum(bad_accel) / len(accel)
    return bad_pct

def rest_acceleration_jitter(speed, time, rest_threshold=0.5):
    """
    Calculate the jitter of acceleration during rest periods.
    """
    accel = np.diff(speed) / np.diff(time)

    # Identify when player is at rest (speed below threshold)
    at_rest = speed < rest_threshold  # m/s threshold for "stationary"

    # Get accelerations during rest periods (align indices)
    rest_accel = accel[at_rest[:-1]]  # [:-1] because accel is 1 shorter than speed

    # Calculate jitter (standard deviation of rest accelerations)
    rest_jitter = np.std(rest_accel)
    return rest_jitter

def path_length_ratio(x, y, x_filt, y_filt):
    """
    Calculate the ratio of path length between raw and filtered data.
    """
    # Calculate distances between consecutive points
    raw_distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    raw_path_length = np.sum(raw_distances)

    # Same for filtered data
    filtered_distances = np.sqrt(np.diff(x_filt)**2 + np.diff(y_filt)**2)
    filtered_path_length = np.sum(filtered_distances)

    # Calculate ratio
    path_length_ratio = filtered_path_length / raw_path_length

    return path_length_ratio

def interpolate_to_uniform(time, speed, target_sample_rate=10):
    """
    Interpolate variable-rate data to uniform sampling rate.
    """
    # Convert to numpy arrays if needed
    time = np.asarray(time)
    speed = np.asarray(speed)
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(time) | np.isnan(speed))
    time = time[valid_mask]
    speed = speed[valid_mask]
    
    # Create uniform time grid
    dt = 1.0 / target_sample_rate  # Time step
    uniform_time = np.arange(time[0], time[-1], dt)
    
    # Interpolate speed to uniform grid
    # Using linear interpolation - could also use 'cubic' for smoother results
    uniform_speed = np.interp(uniform_time, time, speed)
    
    return uniform_time, uniform_speed

def power_spectral_density(time, speed, target_sample_rate=10):
    """
    Calculate the power spectral density of the speed signal.
    """
    uniform_speed = interpolate_to_uniform(time, speed, target_sample_rate)[1]

    # Super simple PSD
    frequencies, psd = signal.welch(uniform_speed, fs=target_sample_rate, nperseg=256)

    # Plot it
    plt.semilogy(frequencies, psd)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectral Density [m²/s²/Hz]')
    plt.title('Speed Signal Frequency Content')