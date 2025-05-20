import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import OSM
from scipy.stats import gaussian_kde
from datetime import datetime

MAP_EXTENT = [-122.6, -121.7, 47.2, 47.5]

df = pd.read_csv('data/query.csv')
# Filter only earthquake data
df = df[df['type'] == 'earthquake']

# Convert time to numeric (days since first event: 1903-03-14)
df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)  # Remove timezone info
first_event = df['time'].min()
df['time_days'] = (df['time'] - first_event).dt.total_seconds() / (24 * 3600)

# Extract earthquake data
eq_times = df['time_days'].to_numpy()
eq_locs = df[['longitude', 'latitude']].to_numpy()
eq_mags = df['mag'].to_numpy()  # Using 'mag' column instead of 'magnitude'

# Calculate mean earthquake location (probable future earthquake spot)
mean_eq_loc = np.mean(eq_locs, axis=0)
print(f"Probable earthquake location: {mean_eq_loc[0]:.4f}°W, {mean_eq_loc[1]:.4f}°N")

# Find the most recent and highest magnitude earthquakes (potential triggers)
recent_high_mag = df.sort_values(['time_days', 'mag'], ascending=[False, False]).head(3)
recent_high_mag_locs = recent_high_mag[['longitude', 'latitude']].to_numpy()
recent_high_mag_mags = recent_high_mag['mag'].to_numpy()



def predict_earthquake_locations(eq_locs, bandwidth=0.01):
    kde = gaussian_kde(eq_locs.T, bw_method=bandwidth)
    west, east, south, north = MAP_EXTENT
    lon_range = np.linspace(west, east, 200)
    lat_range = np.linspace(south, north, 200)
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
    positions = np.vstack([lon_grid.ravel(), lat_grid.ravel()])
    probability = kde(positions).reshape(lon_grid.shape)
    return lon_grid, lat_grid, probability

def compute_etas_intensity(eq_times, eq_locs, eq_mags, current_time):
    west, east, south, north = MAP_EXTENT
    lon_range = np.linspace(west, east, 200)
    lat_range = np.linspace(south, north, 200)
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

def plot_map(eq_locs, lon_grid, lat_grid, intensity, recent_high_mag_locs=None, recent_high_mag_mags=None):
    # Set up the map with OpenStreetMap tiles
    tiler = OSM()
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection=tiler.crs)
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())

    # Add OSM tiles as a base layer
    ax.add_image(tiler, 10)

    # Add geographic features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, edgecolor='blue', alpha=0.5)
    ax.add_feature(cfeature.LAKES, edgecolor='blue', alpha=0.5)

    # Plot aftershock intensity
    intensity_plot = ax.contourf(lon_grid, lat_grid, np.log10(intensity + 1e-10), cmap='Reds', levels=20, transform=ccrs.PlateCarree(), alpha=0.5)
    plt.colorbar(intensity_plot, ax=ax, label='Log10(Aftershock Intensity) (events/day/km²)')

    # Plot earthquake locations
    ax.scatter(eq_locs[:, 0], eq_locs[:, 1], c='blue', s=100, marker='o', label='Earthquakes', transform=ccrs.PlateCarree())

    # Removed Bellevue label as requested
    
    # Plot recent high magnitude earthquakes (potential triggers)
    if recent_high_mag_locs is not None:
        for i, (loc, mag) in enumerate(zip(recent_high_mag_locs, recent_high_mag_mags)):
            size = 200 + mag * 50  # Size based on magnitude
            ax.scatter(loc[0], loc[1], c='orange', s=size, marker='*', 
                      edgecolor='black', linewidth=1,
                      label='Recent High Mag EQ' if i == 0 else '', 
                      transform=ccrs.PlateCarree())
            ax.text(loc[0], loc[1], f'M{mag:.1f}', fontsize=9, 
                   ha='left', va='top', transform=ccrs.PlateCarree())
    
    # Plot simulated M7.0 earthquake (combining with probable location since they're the same)
    ax.scatter(mean_eq_loc[0], mean_eq_loc[1], c='red', s=500, marker='*', 
              edgecolor='black', linewidth=1.5,
              label='Predicted Location of M7 EQ', 
              transform=ccrs.PlateCarree())
    # Removed simulated earthquake label as requested

    plt.title('Probable Earthquake and Aftershock Intensity Map of Seattle Region')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/etas_map.png')
    plt.show()

if __name__ == "__main__":
    # Calculate current time in days since first event to 2025-05-13 22:27 PDT
    current_time = (datetime(2025, 5, 13, 22, 27) - first_event).total_seconds() / (24 * 3600)
    
    # Add simulated M7.0 earthquake at mean location just before current time
    sim_time = current_time - 0.001  # Just before current time
    sim_loc = mean_eq_loc
    sim_mag = 7.0
    
    # Add to earthquake arrays
    eq_times = np.append(eq_times, sim_time)
    eq_locs = np.vstack([eq_locs, sim_loc])
    eq_mags = np.append(eq_mags, sim_mag)
    
    # Compute earthquake probability and ETAS intensity with simulated quake
    lon_grid, lat_grid, intensity = compute_etas_intensity(eq_times, eq_locs, eq_mags, current_time)
    
    # Plot the results with historical earthquakes, simulated quake and probable location
    plot_map(eq_locs, lon_grid, lat_grid, intensity, recent_high_mag_locs, recent_high_mag_mags)