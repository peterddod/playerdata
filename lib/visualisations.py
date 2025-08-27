import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


def visualise_path_comparison(x_raw, y_raw, x_filtered, y_filtered, 
                              segment_raw=None, segment_filtered=None,
                              pitch_x_min=-52.5, pitch_x_max=52.5,
                              pitch_y_min=-34, pitch_y_max=34,
                              player_id="Player X"):
    """
    Visualise the difference between raw and filtered paths on a football pitch.

    Parameters:
    -----------
    x_raw, y_raw: Raw position data
    x_filtered, y_filtered: Filtered position data
    segment_raw, segment_filtered: Optional segment arrays aligned to x/y
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

    # Plot paths (only valid points), breaking across segments if provided
    if segment_raw is not None and np.any(segment_raw > 0):
        first = True
        for seg_id in np.unique(segment_raw[valid_raw]):
            if seg_id <= 0:
                continue
            seg_mask = valid_raw & (segment_raw == seg_id)
            if np.sum(seg_mask) >= 2:
                ax.plot(x_raw[seg_mask], y_raw[seg_mask],
                        'r-', linewidth=0.3, label='Raw (noisy)' if first else None)
                first = False
    else:
        ax.plot(x_raw[valid_raw], y_raw[valid_raw], 
                'r-', linewidth=0.3, label='Raw (noisy)')

    if segment_filtered is not None and np.any(segment_filtered > 0):
        first = True
        for seg_id in np.unique(segment_filtered[valid_filt]):
            if seg_id <= 0:
                continue
            seg_mask = valid_filt & (segment_filtered == seg_id)
            if np.sum(seg_mask) >= 2:
                ax.plot(x_filtered[seg_mask], y_filtered[seg_mask],
                        'b-', linewidth=0.3, label='Filtered' if first else None)
                first = False
    else:
        ax.plot(x_filtered[valid_filt], y_filtered[valid_filt], 
                'b-', linewidth=0.3, label='Filtered')

    # Add start/end markers
    if np.any(valid_raw):
        ax.plot(x_raw[valid_raw][0], y_raw[valid_raw][0], 
                'go', markersize=8, label='Start')
        ax.plot(x_raw[valid_raw][-1], y_raw[valid_raw][-1], 
                'ro', markersize=8, label='End')

    # Calculate distance metrics without connecting across segments if provided
    def _distance_with_optional_segments(x_vals, y_vals, valid_mask, segments):
        if segments is not None and np.any(segments > 0):
            total = 0.0
            for seg_id in np.unique(segments[valid_mask]):
                if seg_id <= 0:
                    continue
                seg_mask = valid_mask & (segments == seg_id)
                if np.sum(seg_mask) >= 2:
                    dx = np.diff(x_vals[seg_mask])
                    dy = np.diff(y_vals[seg_mask])
                    total += float(np.sum(np.sqrt(dx * dx + dy * dy)))
            return total
        dx = np.diff(x_vals[valid_mask])
        dy = np.diff(y_vals[valid_mask])
        return float(np.sum(np.sqrt(dx * dx + dy * dy)))

    raw_dist = _distance_with_optional_segments(x_raw, y_raw, valid_raw, segment_raw)
    filt_dist = _distance_with_optional_segments(x_filtered, y_filtered, valid_filt, segment_filtered)

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


def create_player_heatmap(df, player_id, pitch_x_min=-52.5, pitch_x_max=52.5,
                          pitch_y_min=-34, pitch_y_max=34,
                          grid_cols=25, grid_rows=20,
                          style='discrete', sigma=1.0,
                          cmap='YlOrRd', show_grid=True):
    """
    Generate a heatmap showing time spent in each zone of the pitch.
    
    Parameters:
    -----------
    df : DataFrame
        Data with columns: participation_id, Pitch_x, Pitch_y, time, segment
    player_id : str
        Player identifier to filter data
    pitch_x_min, pitch_x_max : float
        Pitch x boundaries
    pitch_y_min, pitch_y_max : float
        Pitch y boundaries
    grid_cols, grid_rows : int
        Number of columns and rows for heatmap grid
    style : str
        'discrete' for clear cells, 'smooth' for blurred/continuous
    sigma : float
        Gaussian smoothing parameter (only used if style='smooth')
    cmap : str
        Colormap for heatmap
    show_grid : bool
        Show grid lines (only for discrete style)
    
    Returns:
    --------
    fig : matplotlib figure
        The heatmap figure
    """
    # Filter for specific player
    player_data = df[df['participation_id'] == player_id].copy()
    
    if len(player_data) == 0:
        print(f"No data found for player {player_id}")
        return None
    
    # Create grid bins
    x_edges = np.linspace(pitch_x_min, pitch_x_max, grid_cols + 1)
    y_edges = np.linspace(pitch_y_min, pitch_y_max, grid_rows + 1)
    
    # Initialize time grid
    time_grid = np.zeros((grid_rows, grid_cols))
    
    # Process each segment separately to calculate time properly
    if 'segment' in player_data.columns:
        segment_ids = [sid for sid in np.unique(player_data['segment'].values) if sid != 0]
        for segment_id in segment_ids:
            seg_data = player_data[player_data['segment'] == segment_id].sort_values('Time (s)')
            if len(seg_data) < 2:
                continue
            # Calculate time spent at each position
            times = seg_data['Time (s)'].values
            x_vals = seg_data['Pitch_x'].values
            y_vals = seg_data['Pitch_y'].values
            # For each consecutive pair of points
            for i in range(len(seg_data) - 1):
                # Time spent in this position (half before, half after movement)
                time_spent = times[i+1] - times[i]
                # Assign time to both start and end positions (split equally)
                for j, (x, y, weight) in enumerate([(x_vals[i], y_vals[i], 0.5),
                                                    (x_vals[i+1], y_vals[i+1], 0.5)]):
                    x_idx = np.searchsorted(x_edges, x) - 1
                    y_idx = np.searchsorted(y_edges, y) - 1
                    x_idx = np.clip(x_idx, 0, grid_cols - 1)
                    y_idx = np.clip(y_idx, 0, grid_rows - 1)
                    time_grid[y_idx, x_idx] += time_spent * weight
    else:
        # No segment column; treat all data as one continuous run
        seg_data = player_data.sort_values('Time (s)')
        if len(seg_data) >= 2:
            times = seg_data['Time (s)'].values
            x_vals = seg_data['Pitch_x'].values
            y_vals = seg_data['Pitch_y'].values
            for i in range(len(seg_data) - 1):
                time_spent = times[i+1] - times[i]
                for j, (x, y, weight) in enumerate([(x_vals[i], y_vals[i], 0.5),
                                                    (x_vals[i+1], y_vals[i+1], 0.5)]):
                    x_idx = np.searchsorted(x_edges, x) - 1
                    y_idx = np.searchsorted(y_edges, y) - 1
                    x_idx = np.clip(x_idx, 0, grid_cols - 1)
                    y_idx = np.clip(y_idx, 0, grid_rows - 1)
                    time_grid[y_idx, x_idx] += time_spent * weight
    
    # Apply smoothing if requested
    if style == 'smooth':
        time_grid = gaussian_filter(time_grid, sigma=sigma)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 9))
    
    if style == 'discrete':
        # Use pcolormesh for discrete cells with clear boundaries
        X, Y = np.meshgrid(x_edges, y_edges)
        
        if show_grid:
            mesh = ax.pcolormesh(X, Y, time_grid, cmap=cmap, 
                                edgecolors='white', linewidth=0.5,
                                shading='flat')
        else:
            mesh = ax.pcolormesh(X, Y, time_grid, cmap=cmap,
                                shading='flat')
        
        im = mesh  # for colorbar
    else:  # smooth style
        extent = [pitch_x_min, pitch_x_max, pitch_y_min, pitch_y_max]
        im = ax.imshow(time_grid, origin='lower', extent=extent, 
                      cmap=cmap, interpolation='bilinear',
                      aspect='equal', alpha=0.9)
    
    # Draw pitch boundaries
    rect = plt.Rectangle((pitch_x_min, pitch_y_min), 
                         pitch_x_max - pitch_x_min, 
                         pitch_y_max - pitch_y_min,
                         fill=False, edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    
    # Draw center line
    ax.axvline(x=0, color='white', linewidth=1.5, alpha=0.7)
    
    # Draw center circle
    circle = plt.Circle((0, 0), 9.15, fill=False, 
                       edgecolor='white', linewidth=1.5, alpha=0.7)
    ax.add_patch(circle)
    
    # Set labels and title
    ax.set_xlabel('Pitch X (m)', fontsize=12)
    ax.set_ylabel('Pitch Y (m)', fontsize=12)
    
    total_time = np.sum(time_grid)
    title = f'{player_id} - Position Heatmap ({style.capitalize()} Style)\n'
    title += f'Grid: {grid_cols}x{grid_rows} | Total Time: {total_time:.1f}s'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set axis limits
    ax.set_xlim(pitch_x_min, pitch_x_max)
    ax.set_ylim(pitch_y_min, pitch_y_max)
    ax.set_aspect('equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Time in Zone (seconds)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    return fig


def visualise_speed_outliers(speed_original, speed_cleaned, time_original, 
                             time_cleaned, outlier_indices, player_id="Player",
                             segments_original=None, segments_cleaned=None):
    """
    Visualise where speed outliers were removed from the data, accounting for time changes.
    
    Parameters:
    -----------
    speed_original : array-like
        Original speed data with outliers
    speed_cleaned : array-like
        Cleaned speed data with outliers removed/interpolated
    time_original : array-like
        Original time values
    time_cleaned : array-like
        Cleaned time values (may differ due to filtering/interpolation)
    outlier_indices : array-like
        Boolean mask or indices where outliers were detected (for original data)
    player_id : str
        Player identifier for title
    segments_original, segments_cleaned : array-like or None
        Optional segment arrays aligned to the time/speed arrays
    
    Returns:
    --------
    fig : matplotlib figure
        The visualisation figure
    """
    # Convert outlier_indices to boolean mask if needed
    if outlier_indices.dtype != bool:
        mask = np.zeros(len(speed_original), dtype=bool)
        mask[outlier_indices] = True
        outlier_indices = mask
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot original data with optional segment breaks
    if segments_original is not None and np.any(segments_original > 0):
        first = True
        for seg_id in np.unique(segments_original):
            if seg_id <= 0:
                continue
            seg_mask = (segments_original == seg_id)
            ax.plot(time_original[seg_mask], speed_original[seg_mask], 'b-', 
                    label=f'Original Data ({len(time_original)} points)' if first else None)
            first = False
    else:
        ax.plot(time_original, speed_original, 'b-', 
                label=f'Original Data ({len(time_original)} points)')
    
    # Highlight outlier points in red
    ax.scatter(time_original[outlier_indices], speed_original[outlier_indices], 
               color='red', s=30, alpha=0.8, zorder=5,
               label=f'Outliers ({np.sum(outlier_indices)} points)')
    
    # Plot cleaned data (may have different time points) with optional segment breaks
    if segments_cleaned is not None and np.any(segments_cleaned > 0):
        first = True
        for seg_id in np.unique(segments_cleaned):
            if seg_id <= 0:
                continue
            seg_mask = (segments_cleaned == seg_id)
            ax.plot(time_cleaned[seg_mask], speed_cleaned[seg_mask], 'g-', linewidth=1.2,
                    label=f'Cleaned Data ({len(time_cleaned)} points)' if first else None)
            first = False
    else:
        ax.plot(time_cleaned, speed_cleaned, 'g-', linewidth=1.2,
                label=f'Cleaned Data ({len(time_cleaned)} points)')
    
    # Formatting
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Speed (m/s)', fontsize=12)
    
    # Calculate statistics
    outlier_count = np.sum(outlier_indices)
    outlier_pct = 100 * np.mean(outlier_indices)
    points_removed = len(time_original) - len(time_cleaned)
    
    title = f'{player_id} - Speed Outlier Removal\n'
    if points_removed > 0:
        title += f'Removed {outlier_count} outliers ({outlier_pct:.1f}%) | '
        title += f'Total points: {len(time_original)} → {len(time_cleaned)}'
    else:
        title += f'Corrected {outlier_count} outliers ({outlier_pct:.1f}%) via interpolation'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add grid for readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Set x-axis limits to show full time range
    x_min = min(np.min(time_original), np.min(time_cleaned)) if len(time_cleaned) > 0 else np.min(time_original)
    x_max = max(np.max(time_original), np.max(time_cleaned)) if len(time_cleaned) > 0 else np.max(time_original)
    ax.set_xlim(x_min - 1, x_max + 1)
    
    plt.tight_layout()
    return fig