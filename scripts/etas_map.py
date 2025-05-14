import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import OSM
from scipy.stats import gaussian_kde
from datetime import datetime

# Read the earthquake data
df = pd.read_csv('data/Earthquakes in Seattle.csv')
# Filter only earthquake data
df = df[df['type'] == 'earthquake']

# Convert time to numeric (days since first event: 1903-03-14)
df['time'] = pd.to_datetime(df['time'])
first_event = df['time'].min()
df['time_days'] = (df['time'] - first_event).dt.total_seconds() / (24 * 3600)

# Extract earthquake data
eq_times = df['time_days'].to_numpy()
eq_locs = df[['longitude', 'latitude']].to_numpy()
eq_mags = df['magnitude'].to_numpy()

# Calculate mean earthquake location
mean_eq_loc = np.mean(eq_locs, axis=0)
print(f"Mean earthquake location: {mean_eq_loc[0]:.4f}°W, {mean_eq_loc[1]:.4f}°N")

# Read station data (assuming synthetic data from previous CSV)
station_data = pd.read_csv('data/seismic_data.csv')
station_data = station_data[station_data['type'] == 'station']
stations = station_data[['longitude', 'latitude']].to_numpy()
profile_names = station_data['profile_name'].to_numpy()

def predict_earthquake_locations(eq_locs, bandwidth=0.01):
    kde = gaussian_kde(eq_locs.T, bw_method=bandwidth)
    lon_range = np.linspace(-122.5, -122.1, 200)
    lat_range = np.linspace(47.4, 47.7, 200)
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
    positions = np.vstack([lon_grid.ravel(), lat_grid.ravel()])
    probability = kde(positions).reshape(lon_grid.shape)
    return lon_grid, lat_grid, probability

def compute_etas_intensity(eq_times, eq_locs, eq_mags, current_time):
    lon_range = np.linspace(-122.5, -122.1, 200)
    lat_range = np.linspace(47.4, 47.7, 200)
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)

    # ETAS parameters
    mu = 0.01  # background rate
    k = 0.1    # aftershock productivity
    alpha = 1.0  # magnitude scaling
    c = 0.01   # time offset
    p = 1.0    # temporal decay
    d = 0.1    # spatial scaling
    q = 2.0    # spatial decay
    M0 = 4.0   # reference magnitude

    # Get predicted earthquake probability
    _, _, eq_probability = predict_earthquake_locations(eq_locs)
    intensity = np.full_like(lon_grid, mu) * eq_probability
    grid_points = np.stack([lon_grid.ravel(), lat_grid.ravel()], axis=1)

    for t_i, loc_i, M_i in zip(eq_times, eq_locs, eq_mags):
        if t_i < current_time:
            dt = current_time - t_i + c
            r = np.sqrt(((grid_points[:, 0] - loc_i[0]) * 111)**2 + 
                        ((grid_points[:, 1] - loc_i[1]) * 111 * np.cos(np.radians(loc_i[1])))**2) + d
            productivity = k * np.exp(alpha * (M_i - M0))
            temporal_term = productivity / (dt ** p)
            spatial_term = 1 / (r ** q)
            intensity += (temporal_term * spatial_term).reshape(lon_grid.shape)

    return lon_grid, lat_grid, intensity

def plot_map(eq_locs, stations, profile_names, lon_grid, lat_grid, intensity):
    # Set up the map with OpenStreetMap tiles
    tiler = OSM()
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection=tiler.crs)
    ax.set_extent([-122.5, -122.1, 47.4, 47.7], crs=ccrs.PlateCarree())

    # Add OSM tiles as a base layer
    ax.add_image(tiler, 10)

    # Add geographic features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, edgecolor='blue', alpha=0.5)
    ax.add_feature(cfeature.LAKES, edgecolor='blue', alpha=0.5)

    # Plot aftershock intensity
    intensity_plot = ax.contourf(lon_grid, lat_grid, np.log10(intensity + 1e-10), cmap='Reds', levels=20, transform=ccrs.PlateCarree())
    plt.colorbar(intensity_plot, ax=ax, label='Log10(Aftershock Intensity) (events/day/km²)')

    # Plot earthquake locations
    ax.scatter(eq_locs[:, 0], eq_locs[:, 1], c='blue', s=100, marker='o', label='Earthquakes', transform=ccrs.PlateCarree())

    # Plot mean earthquake location
    ax.scatter(mean_eq_loc[0], mean_eq_loc[1], c='yellow', s=400, marker='*', label='Mean EQ Location', transform=ccrs.PlateCarree())
    ax.text(mean_eq_loc[0], mean_eq_loc[1], 'Mean EQ Location', fontsize=10, ha='right', va='bottom', transform=ccrs.PlateCarree())

    # Plot seismic profiles
    unique_profiles = np.unique(profile_names)
    colors = ['green', 'purple', 'orange']
    for idx, profile in enumerate(unique_profiles):
        profile_stations = stations[profile_names == profile]
        ax.plot(profile_stations[:, 0], profile_stations[:, 1], c=colors[idx], linewidth=2, label=profile, transform=ccrs.PlateCarree())
        mid_idx = len(profile_stations) // 2
        ax.text(profile_stations[mid_idx, 0], profile_stations[mid_idx, 1], profile, 
                transform=ccrs.PlateCarree(), fontsize=8, ha='right', color=colors[idx])

    # Add recognizable places
    places = {
        'Seattle': (-122.33, 47.60),
        'Bellevue': (-122.20, 47.61),
        'Mercer Island': (-122.22, 47.57),
        'Renton': (-122.21, 47.48),
        'Lake Washington': (-122.26, 47.62)  # Approximate center
    }
    for place, (lon, lat) in places.items():
        ax.text(lon, lat, place, transform=ccrs.PlateCarree(), fontsize=10, ha='left', va='bottom', color='black', weight='bold')

    plt.title('ETAS Aftershock Intensity Map of Seattle Region')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/etas_map.png')
    plt.show()

if __name__ == "__main__":
    # Calculate current time in days since first event (1903-03-14 to 2025-05-13 22:27 PDT)
    current_time = (datetime(2025, 5, 13, 22, 27) - first_event).total_seconds() / (24 * 3600)
    
    # Compute earthquake probability and ETAS intensity
    lon_grid, lat_grid, intensity = compute_etas_intensity(eq_times, eq_locs, eq_mags, current_time)
    
    # Plot the results
    plot_map(eq_locs, stations, profile_names, lon_grid, lat_grid, intensity)