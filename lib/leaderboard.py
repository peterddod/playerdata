import numpy as np

def zone5_distance(speed, x, y):
    # Zone 5 speed range: 19.8 km/h to 25.1 km/h
    # Convert to m/s: 19.8 km/h = 5.5 m/s, 25.1 km/h = 6.97 m/s
    # Find indices where speed is in zone 5
    zone5_mask = (speed[:-1] >= 5.5) & (speed[:-1] < 6.97)
    # Calculate distance only for those segments
    dx = np.diff(x)
    dy = np.diff(y)
    zone5_distances = np.sqrt(dx**2 + dy**2)[zone5_mask]
    zone5_distance = np.sum(zone5_distances)

    return zone5_distance

def total_distance(x, y):
    return np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

def max_speed(speed):
    return np.max(speed)