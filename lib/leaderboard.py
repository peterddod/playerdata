import numpy as np

def total_distance_segmented(df, player_id):
    """
    Calculate total distance for a player, not counting jumps between segments.
    """
    player_data = df[df['participation_id'] == player_id]
    if len(player_data) == 0:
        return 0

    player_data = player_data.sort_values(['Time (s)'])

    # Vectorized per-segment diffs
    dx = player_data.groupby('segment')['Pitch_x'].diff()
    dy = player_data.groupby('segment')['Pitch_y'].diff()
    distances = np.sqrt(dx * dx + dy * dy)

    # Sum distances within segments (ignore NaN at segment starts)
    return float(np.nansum(distances))

def speed_filter(df, player_id=None, min_speed=5.5, max_speed=6.97):
    """
    Filter to rows where speed is within [min_speed, max_speed) and
    re-number contiguous in-range runs as new segments, respecting existing
    segments in the input DataFrame.

    Rules:
    - Contiguous in-range rows within the same original segment form one new segment.
    - New segment IDs are assigned sequentially starting at 0 across time,
      not across original segment boundaries (i.e., numbering continues across
      original segments in chronological order, but runs never bridge them).

    Parameters
    ----------
    df : DataFrame
        Player data, or the full dataset. Must contain columns:
        'participation_id', 'Time (s)', 'Speed (m/s)', and 'segment'.
    player_id : str or None
        If provided, filter df to this player; otherwise df is assumed to be
        already filtered to a single player.
    min_speed, max_speed : float
        Inclusive lower bound and exclusive upper bound for speed filtering.

    Returns
    -------
    DataFrame
        Filtered DataFrame containing only in-range rows with a re-assigned
        integer 'segment' column as described above.
    """
    # Scope to single player if requested
    player_data = df[df['participation_id'] == player_id].copy() if player_id is not None else df.copy()

    if len(player_data) == 0:
        return player_data.iloc[0:0].copy()

    # Ensure required columns exist
    required_cols = {'Time (s)', 'Speed (m/s)', 'segment'}
    missing = required_cols - set(player_data.columns)
    if missing:
        raise KeyError(f"Missing required columns for speed_filter: {sorted(missing)}")

    # Sort for deterministic processing; original segments are respected
    player_data = player_data.sort_values(['Time (s)'])

    # In-range mask
    in_range = (player_data['Speed (m/s)'] >= min_speed) & (player_data['Speed (m/s)'] < max_speed)

    # Group by original segments to avoid bridging across them
    original_segments = player_data['segment'].astype(int)

    # Detect run starts within each original segment
    entered_run = in_range & ~in_range.groupby(original_segments).shift(fill_value=False)

    # Local run id within each original segment (0-based)
    local_run_id = entered_run.groupby(original_segments).cumsum() - 1

    # Compute offsets so that numbering continues across original segments in time order
    starts_per_orig_segment = entered_run.groupby(original_segments).sum().astype(int)
    segment_min_time = player_data.groupby(original_segments)['Time (s)'].min()
    ordered_segments = segment_min_time.sort_values().index.tolist()

    offset_map = {}
    current_offset = 0
    for seg in ordered_segments:
        offset_map[seg] = current_offset
        current_offset += int(starts_per_orig_segment.get(seg, 0))

    global_run_id = local_run_id + original_segments.map(offset_map)

    # Build filtered output with re-numbered segments
    filtered = player_data[in_range].copy()
    if len(filtered) == 0:
        return filtered
    filtered['segment'] = global_run_id[in_range].astype(int)

    return filtered

def max_speed(df, player_id):
    """
    Get max speed for a player.
    """
    player_data = df[df['participation_id'] == player_id]
    
    if len(player_data) == 0:
        return 0
    
    return player_data['Speed (m/s)'].max()