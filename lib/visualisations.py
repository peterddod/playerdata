import matplotlib.pyplot as plt
import numpy as np

def visualise_path_comparison(x_raw, y_raw, x_filtered, y_filtered, 
                              pitch_x_min=-52.5, pitch_x_max=52.5,
                              pitch_y_min=-34, pitch_y_max=34,
                              player_id="Player X"):
    """
    Visualise the difference between raw and filtered paths on a football pitch.

    Parameters:
    -----------
    x_raw, y_raw: Raw position data
    x_filtered, y_filtered: Filtered position data
    pitch_x_min, pitch_x_max: Pitch x boundaries
    pitch_y_min, pitch_y_max: Pitch y boundaries
    player_id: Player identifier for title
    """
    # Filter data to only include points within pitch
    valid_raw = (x_raw >= pitch_x_min) & (x_raw <= pitch_x_max) & \
                (y_raw >= pitch_y_min) & (y_raw <= pitch_y_max)
    valid_filt = (x_filtered >= pitch_x_min) & (x_filtered <= pitch_x_max) & \
                 (y_filtered >= pitch_y_min) & (y_filtered <= pitch_y_max)

    # Create figure and use current axes (no subplots)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Set pitch limits
    ax.set_xlim(pitch_x_min, pitch_x_max)
    ax.set_ylim(pitch_y_min, pitch_y_max)
    ax.set_aspect('equal')

    # Draw pitch boundaries
    rect = plt.Rectangle((pitch_x_min, pitch_y_min), 
                         pitch_x_max - pitch_x_min, 
                         pitch_y_max - pitch_y_min,
                         fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(rect)

    # Plot paths (only valid points)
    ax.plot(x_raw[valid_raw], y_raw[valid_raw], 
            'r-', linewidth=0.3, label='Raw (noisy)')
    ax.plot(x_filtered[valid_filt], y_filtered[valid_filt], 
            'b-', linewidth=0.3, label='Filtered')

    # Add start/end markers
    if np.any(valid_raw):
        ax.plot(x_raw[valid_raw][0], y_raw[valid_raw][0], 
                'go', markersize=8, label='Start')
        ax.plot(x_raw[valid_raw][-1], y_raw[valid_raw][-1], 
                'ro', markersize=8, label='End')

    # Calculate distance metrics
    raw_dist = np.sum(np.sqrt(np.diff(x_raw[valid_raw])**2 + 
                              np.diff(y_raw[valid_raw])**2))
    filt_dist = np.sum(np.sqrt(np.diff(x_filtered[valid_filt])**2 + 
                               np.diff(y_filtered[valid_filt])**2))

    ax.set_xlabel('Pitch X (m)', fontsize=12)
    ax.set_ylabel('Pitch Y (m)', fontsize=12)
    ax.set_title(f'{player_id} - Path Comparison\n'
                 f'Raw Distance: {raw_dist:.1f}m | '
                 f'Filtered Distance: {filt_dist:.1f}m | '
                 f'Reduction: {raw_dist - filt_dist:.1f}m ({100*(1-filt_dist/raw_dist):.1f}%)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig