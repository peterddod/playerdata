import numpy as np
from scipy import signal


def physical_limits_outlier_removal(speed, time, max_speed=12, max_accel=8, max_decel=-9):
    """
    Remove physically impossible values and interpolate.
    """
    # Calculate acceleration
    dt = np.diff(time)
    accel = np.diff(speed) / dt
    
    # Flag outliers
    speed_outliers = (speed > max_speed) | (speed < 0)
    
    # Acceleration outliers (padding to match speed array length)
    accel_outliers = np.concatenate([
        [False],  # First point has no acceleration
        (accel > max_accel) | (accel < max_decel)
    ])
    
    # Combine outlier flags
    outliers = speed_outliers | accel_outliers
    
    # Interpolate outliers using linear interpolation
    valid_points = ~outliers
    if np.any(outliers) and np.sum(valid_points) > 1:
        speed_cleaned = np.copy(speed)
        speed_cleaned[outliers] = np.interp(
            time[outliers],      # Where to interpolate
            time[valid_points],  # Known good points
            speed[valid_points]  # Known good values
        )
    else:
        speed_cleaned = speed
    
    return speed_cleaned, outliers

def filter_to_pitch_boundaries_with_segments(df, pitch_x_min, pitch_x_max, 
                                              pitch_y_min, pitch_y_max):
    """
    Filter dataframe to pitch boundaries and add segment tracking for each player.
    """
    # Identify valid points
    valid = (df['Pitch_x'] >= pitch_x_min) & (df['Pitch_x'] <= pitch_x_max) & \
            (df['Pitch_y'] >= pitch_y_min) & (df['Pitch_y'] <= pitch_y_max)

    # Vectorized segment assignment per player:
    # - Detect first valid sample after any invalid run within each player
    # - Cumulatively count these entries to form segment IDs
    group_key = df['participation_id']
    entered_segment = valid & ~valid.groupby(group_key).shift(fill_value=False)
    segment_ids = entered_segment.groupby(group_key).cumsum()

    # Filter to only valid points and assign segment ids on the copy
    df_filtered = df.loc[valid].copy()
    df_filtered['segment'] = segment_ids.loc[valid].astype(int)

    return df_filtered

def interpolate_to_uniform(time, speed, pitch_x, pitch_y, target_sample_rate=10):
    """
    Interpolate all signals to uniform sampling rate.
    """
    time = np.asarray(time)
    speed = np.asarray(speed)
    pitch_x = np.asarray(pitch_x)
    pitch_y = np.asarray(pitch_y)
    
    # Remove NaN values (from ANY column)
    valid_mask = ~(np.isnan(time) | np.isnan(speed) | 
                   np.isnan(pitch_x) | np.isnan(pitch_y))
    time = time[valid_mask]
    speed = speed[valid_mask]
    pitch_x = pitch_x[valid_mask]
    pitch_y = pitch_y[valid_mask]
    
    # Create uniform time grid
    dt = 1.0 / target_sample_rate
    uniform_time = np.arange(time[0], time[-1], dt)
    
    # Interpolate ALL signals to same grid
    uniform_speed = np.interp(uniform_time, time, speed)
    uniform_x = np.interp(uniform_time, time, pitch_x)
    uniform_y = np.interp(uniform_time, time, pitch_y)
    
    return uniform_time, uniform_speed, uniform_x, uniform_y

def butterworth_filter(speed, cutoff_freq=2.0, fs=10, order=4):
    """
    Apply zero-phase Butterworth low-pass filter to speed data.
    
    Parameters:
    - speed: Speed signal
    - cutoff_freq: Cutoff frequency in Hz (default 2 Hz)
    - fs: Sampling frequency in Hz (default 10 Hz)
    - order: Filter order (default 4)
    """
    # Normalize cutoff frequency (Nyquist = fs/2)
    nyquist = fs / 2
    normal_cutoff = cutoff_freq / nyquist
    
    # Design Butterworth filter
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply filter forward and backward (zero-phase)
    speed_filtered = signal.filtfilt(b, a, speed)
    
    return speed_filtered

def savgol_filter(x, y, window_length=7, poly_order=2):
    """
    Apply Savitzky-Golay filter to smooth position data.
    
    Parameters:
    - x, y: Position coordinates
    - window_length: Number of points in filter window (must be odd)
    - poly_order: Polynomial order (must be < window_length)
    """
    # Apply filter to each coordinate separately
    x_smooth = signal.savgol_filter(x, window_length, poly_order)
    y_smooth = signal.savgol_filter(y, window_length, poly_order)
    
    return x_smooth, y_smooth