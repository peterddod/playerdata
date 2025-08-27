# Process for Getting Data Results

## Phase 1: Data cleaning
1. Remove outliers in speed based on physical limits and replace with interpolated values
2. Run zero‑phase butterworth low‑pass filter on speed
3. Run savgol filter on positions
4. Remove datapoints outside of the pitch

## Phase 2: Generate leaderboards
Use cleaned data to generate leaderboards for:
- Total distance
- Distance at Zone 5
- Top speed

## Phase 3: Produce visuals
1. Create heatmap for player positions
2. Create zonal heatmap for players i.e. where they spend most time at rest, where they spend most time sprinting
3. Create on-ball and off-ball position chart

For each of these, use a couple of players to showcase different examples of player characteristics they show.

# Tasks to complete
- [x] Create a test function to determine level of noise in dataset
- [x] Outlier removal function
- [x] Butterworth filter function
- [x] Savgol filter function
- [x] Boundary filtering function
- [x] Implement in ipynb

- [x] Leaderboards
- [x] Heatmap
- [x] Zonal heatmap
- [x] On/Off-Ball position chart