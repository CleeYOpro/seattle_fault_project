import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import OSM
from scipy.stats import gaussian_kde
from datetime import datetime
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.lines as mlines

MAP_EXTENT = [-122.8, -121.6, 47.2, 47.9]

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
    lon_range = np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 200)
    lat_range = np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 200)
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)

    mu, k, alpha, c, p, d, q, M0 = 0.01, 0.1, 1.0, 0.01, 1.0, 0.1, 2.0, 4.0
    _, _, eq_probability = predict_earthquake_locations(eq_locs)
    intensity = np.full_like(lon_grid, mu) * eq_probability
    grid_points = np.stack([lon_grid.ravel(), lat_grid.ravel()], axis=1)

    for t_i, loc_i, M_i in zip(eq_times, eq_locs, eq_mags):
        if t_i < current_time:
            dt = current_time - t_i + c
            r = np.sqrt(((grid_points[:, 0] - loc_i[0]) * 111)**2 + ((grid_points[:, 1] - loc_i[1]) * 111 * np.cos(np.radians(loc_i[1])))**2) + d
            productivity = k * np.exp(alpha * (M_i - M0))
            temporal_term = productivity / (dt ** p)
            spatial_term = 1 / (r ** q)
            intensity += (temporal_term * spatial_term).reshape(lon_grid.shape)
    return lon_grid, lat_grid, intensity

def calculate_mmi(magnitude, distance_km):
    """
    Calculate Modified Mercalli Intensity (MMI) based on magnitude and distance.
    Uses the Atkinson and Wald (2007) relationship.
    
    Args:
        magnitude: Earthquake magnitude
        distance_km: Distance from epicenter in kilometers
        
    Returns:
        Estimated MMI value (1-12 scale)
    """
    # Calculate PGA in g units using a simplified GMPE for crustal earthquakes
    # Based on simplified Boore-Atkinson (2008) model
    log_pga = 0.3 + 0.59 * (magnitude - 6) - 0.0075 * distance_km - 1.6 * np.log10(distance_km + 10)
    pga = 10 ** log_pga
    
    # Convert PGA to MMI using Worden et al. (2012) with vectorized operations
    mmi = np.where(pga < 0.0017, 1.0, 3.66 * np.log10(pga * 980) - 1.66)
    
    # Clip MMI values to valid range [1, 12]
    mmi = np.clip(mmi, 1, 12)
    
    return mmi

def compute_mmi_grid(epicenter, magnitude, lon_grid, lat_grid):
    """
    Compute MMI values for a grid based on distance from epicenter.
    
    Args:
        epicenter: Tuple of (longitude, latitude) for earthquake epicenter
        magnitude: Earthquake magnitude
        lon_grid: Grid of longitude values
        lat_grid: Grid of latitude values
        
    Returns:
        Grid of MMI values
    """
    # Calculate distance from each grid point to epicenter in km
    # Using approximate conversion: 1 degree ≈ 111 km
    distances = np.sqrt(
        ((lon_grid - epicenter[0]) * 111 * np.cos(np.radians(epicenter[1]))) ** 2 + 
        ((lat_grid - epicenter[1]) * 111) ** 2
    )
    
    # Calculate MMI for each point
    mmi_grid = calculate_mmi(magnitude, distances)
    
    return mmi_grid

def plot_map_mmi(eq_locs, lon_grid, lat_grid, mmi_grid, recent_high_mag_locs=None, recent_high_mag_mags=None):
    # Set up the map with OpenStreetMap tiles
    tiler = OSM()
    # Calculate aspect ratio based on MAP_EXTENT
    width = MAP_EXTENT[1] - MAP_EXTENT[0]
    height = MAP_EXTENT[3] - MAP_EXTENT[2]
    aspect_ratio = width / height
    fig = plt.figure(figsize=(10 * aspect_ratio, 10))
    ax = fig.add_subplot(111, projection=tiler.crs)
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_image(tiler, 10)

    # Add geographic features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, edgecolor='blue', alpha=0.5)
    ax.add_feature(cfeature.LAKES, edgecolor='blue', alpha=0.5)

    # Define MMI color map
    mmi_colors = {
        1: '#FFFFFF',  # I - Not felt (White)
        2: '#ACD8E9',  # II - Very light (Light blue)
        3: '#83D0DA',  # III - Light (Aquamarine)
        4: '#7BC87F',  # IV - Moderate (Light green)
        5: '#F9F518',  # V - Rather strong (Yellow)
        6: '#FAC611',  # VI - Strong (Light orange)
        7: '#FA8A11',  # VII - Very strong (Orange)
        8: '#F7100C',  # VIII - Destructive (Red)
        9: '#C80F0A',  # IX - Violent (Dark red)
        10: '#A80C09', # X - Intense (Darker red)
        11: '#800B0D', # XI - Extreme (Very dark red)
        12: '#590B0D'  # XII - Catastrophic (Maroon)
    }
    
    # Create custom colormap for MMI
    mmi_levels = np.arange(1, 13)
    colors = [mmi_colors[i] for i in mmi_levels]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(mmi_levels, cmap.N)
    
    # Plot MMI contours
    mmi_plot = ax.contourf(lon_grid, lat_grid, mmi_grid, 
                          levels=mmi_levels, 
                          cmap=cmap, 
                          norm=norm,
                          transform=ccrs.PlateCarree(), 
                          alpha=0.7)
    
    # Create a custom legend for MMI scale
    mmi_descriptions = {
        1: 'I - Not felt',
        2: 'II - Weak',
        3: 'III - Light',
        4: 'IV - Moderate',
        5: 'V - Rather Strong',
        6: 'VI - Strong',
        7: 'VII - Very Strong',
        8: 'VIII - Destructive',
        9: 'IX - Violent',
        10: 'X - Intense',
        11: 'XI - Extreme',
        12: 'XII - Catastrophic'
    }
    
    # Create legend patches for MMI scale
    mmi_patches = []
    for i in range(1, 13):
        if i in [1, 3, 5, 7, 9, 11]:  # Show only odd-numbered levels to save space
            mmi_patches.append(Patch(color=mmi_colors[i], label=mmi_descriptions[i]))
    
    # Add colorbar
    cbar = plt.colorbar(mmi_plot, ax=ax, shrink=0.7, pad=0.05)
    cbar.set_label('Modified Mercalli Intensity Scale')
    
    # Plot earthquake locations
    historical_eq = ax.scatter(eq_locs[:, 0], eq_locs[:, 1], c='blue', s=10, marker='o', 
                              label='Historical Earthquakes (1903-2025)', 
                              transform=ccrs.PlateCarree())
    
    # Plot recent high magnitude earthquakes (potential triggers)
    recent_eq_handle = None
    if recent_high_mag_locs is not None:
        for i, (loc, mag) in enumerate(zip(recent_high_mag_locs, recent_high_mag_mags)):
            size = 200 + mag * 50  # Size based on magnitude
            ax.scatter(loc[0], loc[1], c='orange', s=size, marker='*', 
                      edgecolor='black', linewidth=1,
                      label='Recent High Mag EQ' if i == 0 else '', 
                      transform=ccrs.PlateCarree())
            ax.text(loc[0], loc[1], f'M{mag:.1f}', fontsize=9, 
                   ha='left', va='top', transform=ccrs.PlateCarree())
            if i == 0:
                recent_eq_handle = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='orange', 
                                             markersize=15, label='Recent High Mag EQ')
    
    # Plot simulated M7.0 earthquake
    sim_eq = ax.scatter(mean_eq_loc[0], mean_eq_loc[1], c='red', s=300, marker='*', 
                        edgecolor='black', linewidth=1.5,
                        label='Predicted Location of M7 EQ', 
                        transform=ccrs.PlateCarree())
    
    # Combine MMI patches with earthquake markers for legend
    handles = mmi_patches + [historical_eq, sim_eq]
    if recent_eq_handle is not None:
        handles.append(recent_eq_handle)
    
    plt.title('Modified Mercalli Intensity (MMI) Map for AfterShocks of Simulated M7.0 Earthquake')
    plt.legend(handles=handles, loc='lower left', fontsize=8)
    plt.tight_layout()
    plt.savefig('outputs/mmi_map.png')
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
    
    # Create grid for plotting
    west, east, south, north = MAP_EXTENT
    lon_range = np.linspace(west, east, 200)
    lat_range = np.linspace(south, north, 200)
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
    
    # Compute MMI grid for the simulated M7.0 earthquake
    mmi_grid = compute_mmi_grid(sim_loc, sim_mag, lon_grid, lat_grid)
    
    # Plot the results with historical earthquakes, simulated quake and MMI intensity
    plot_map_mmi(eq_locs, lon_grid, lat_grid, mmi_grid, recent_high_mag_locs, recent_high_mag_mags)


    aftershock_mag = ETAS_return
    distances = np.sqrt(((lon_grid - mean_eq_loc[0]) * 111 * np.cos(np.radians(mean_eq_loc[1])))**2 + ((lat_grid - mean_eq_loc[1]) * 111)**2)
    log_pga = 0.3 + 0.59 * (aftershock_mag - 6) - 0.0075 * distances - 1.6 * np.log10(distances + 10)
    pga = 10 ** log_pga
    mmi = np.where(pga < 0.0017, 1.0, 3.66 * np.log10(pga * 980) - 1.66)
    mmi = np.clip(mmi, 1, 12)
    weighted_mmi = mmi * intensity / np.max(intensity)  # Scale MMI by aftershock probability