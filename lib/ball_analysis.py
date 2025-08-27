import numpy as np
import pandas as pd


def proximity_to_ball(df, player_id, ball_id, tolerance=0.2):
    """
    Compute the player's distance to the ball for each player datapoint.

    For each player timestamp, find the ball sample with the nearest time
    within the given tolerance and compute Euclidean distance on Pitch_x/y.
    If no ball sample exists within the tolerance, the distance is NaN.

    Parameters
    ----------
    df : DataFrame
        Dataset containing at least 'participation_id', 'Time (s)', 'Pitch_x', 'Pitch_y'.
    player_id : str
        Participation id for the player.
    ball_id : str
        Participation id for the ball (e.g., 'ball').
    tolerance : float
        Maximum allowed absolute time difference (seconds) for nearest matching.

    Returns
    -------
    DataFrame
        A DataFrame aligned to the player's timestamps with columns:
        ['participation_id', 'Time (s)', ('segment' if present), 'distance_to_ball'].
        Distances are NaN where no ball sample is within tolerance.
    """
    required_cols = {'participation_id', 'Time (s)', 'Pitch_x', 'Pitch_y'}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for proximity_to_ball: {sorted(missing)}")

    # Extract player and ball tracks
    player = df[df['participation_id'] == player_id].copy()
    ball = df[df['participation_id'] == ball_id].copy()

    if len(player) == 0 or len(ball) == 0:
        return pd.DataFrame(columns=['participation_id', 'Time (s)', 'distance_to_ball'])

    # Sort for merge_asof
    player = player.sort_values('Time (s)')
    ball = ball.sort_values('Time (s)')

    # As-of nearest merge within tolerance
    merged = pd.merge_asof(
        player,
        ball[['Time (s)', 'Pitch_x', 'Pitch_y']],
        on='Time (s)',
        direction='nearest',
        tolerance=tolerance,
        suffixes=('', '_ball')
    )

    # Compute distance where ball match exists (non-NaN)
    dx = merged['Pitch_x'] - merged['Pitch_x_ball']
    dy = merged['Pitch_y'] - merged['Pitch_y_ball']
    distance = np.sqrt(dx * dx + dy * dy)

    # Prepare result aligned to player times
    cols = ['participation_id', 'Time (s)', 'Pitch_x', 'Pitch_y']
    if 'segment' in player.columns:
        cols.append('segment')
    result = merged[cols].copy()
    result['distance_to_ball'] = distance

    return result


def filter_player_by_ball_proximity(df, player_id, ball_id, max_distance, tolerance=0.2):
    """
    Filter a single player's rows to those within max_distance of the ball,
    re-segmenting contiguous in-range runs while respecting original segments.

    Parameters
    ----------
    df : DataFrame
        Dataset with columns 'participation_id', 'Time (s)', 'Pitch_x', 'Pitch_y', 'segment'.
    player_id : str
        Player participation id.
    ball_id : str
        Ball participation id (e.g., 'ball').
    max_distance : float
        Maximum allowed player-to-ball distance (meters).
    tolerance : float
        Nearest-time matching tolerance in seconds when pairing with ball samples.

    Returns
    -------
    DataFrame
        Filtered player DataFrame with updated integer 'segment' and an
        added 'distance_to_ball' column. Segment IDs are 0..N-1 across time
        and never bridge original segment boundaries.
    """
    required_cols = {'participation_id', 'Time (s)', 'Pitch_x', 'Pitch_y', 'segment'}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for filter_player_by_ball_proximity: {sorted(missing)}")

    # Compute per-point distances to the ball aligned to player timestamps
    dist_df = proximity_to_ball(df, player_id=player_id, ball_id=ball_id, tolerance=tolerance)

    # Extract the player's full rows and align distances
    player = df[df['participation_id'] == player_id].copy().sort_values('Time (s)')
    player = player.merge(
        dist_df[['Time (s)', 'distance_to_ball']],
        on='Time (s)', how='left'
    )

    # Keep only rows within the distance threshold (and with a matched ball sample)
    in_range = (player['distance_to_ball'].notna()) & (player['distance_to_ball'] <= max_distance)
    if not in_range.any():
        return player.iloc[0:0].copy()

    # Respect original segments when forming contiguous runs
    original_segments = player['segment'].astype(int)
    entered_run = in_range & ~in_range.groupby(original_segments).shift(fill_value=False)
    local_run_id = entered_run.groupby(original_segments).cumsum() - 1

    # Compute global offsets so segment numbering is sequential across time
    starts_per_orig_segment = entered_run.groupby(original_segments).sum().astype(int)
    segment_min_time = player.groupby(original_segments)['Time (s)'].min()
    ordered_segments = segment_min_time.sort_values().index.tolist()

    offset_map = {}
    current_offset = 0
    for seg in ordered_segments:
        offset_map[seg] = current_offset
        current_offset += int(starts_per_orig_segment.get(seg, 0))

    global_run_id = local_run_id + original_segments.map(offset_map)

    # Build filtered output with re-numbered segments
    filtered = player[in_range].copy()
    filtered['segment'] = global_run_id[in_range].astype(int)

    return filtered