import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

df = pd.read_csv('data/seismic_data.csv')

eq_data = df[df['type'] == 'earthquake']
eq_times = eq_data['time'].to_numpy()
eq_locs = eq_data[['longitude', 'latitude']].to_numpy()
eq_mags = eq_data['magnitude'].to_numpy()

station_data = df[df['type'] == 'station']
stations = station_data[['longitude', 'latitude']].to_numpy()
profile_names = station_data['profile_name'].to_numpy()

def compute_etas_intensity(eq_times, eq_locs, eq_mags, current_time):
    lon_range = np.linspace(-122.4229, -122.1865, 200)
    lat_range = np.linspace(47.5095, 47.60116, 200)
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)

    mu = 0.01
    k = 0.1
    alpha = 1.0
    c = 0.01
    p = 1.0
    d = 0.1
    q = 2.0
    M0 = 4.0

    intensity = np.full_like(lon_grid, mu)
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
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-122.4229, -122.1865, 47.5095, 47.60116], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True)

    plot = ax.contourf(lon_grid, lat_grid, np.log10(intensity + 1e-10), cmap='Reds', levels=20, transform=ccrs.PlateCarree())
    plt.colorbar(plot, ax=ax, label='Log10(Aftershock Intensity)')

    ax.scatter(eq_locs[:, 0], eq_locs[:, 1], c='blue', s=100, marker='o', label='Earthquakes', transform=ccrs.PlateCarree())

    unique_profiles = np.unique(profile_names)
    colors = ['green', 'purple', 'orange']
    for idx, profile in enumerate(unique_profiles):
        profile_stations = stations[profile_names == profile]
        ax.plot(profile_stations[:, 0], profile_stations[:, 1], c=colors[idx], linewidth=2, label=profile, transform=ccrs.PlateCarree())
        mid_idx = len(profile_stations) // 2
        ax.text(profile_stations[mid_idx, 0], profile_stations[mid_idx, 1], profile, 
                transform=ccrs.PlateCarree(), fontsize=8, ha='right', color=colors[idx])

    plt.title('ETAS Aftershock Intensity and Seismic Profiles')
    plt.legend()
    plt.savefig('outputs/etas_map.png')
    plt.show()

if __name__ == "__main__":
    current_time = 30.0
    lon_grid, lat_grid, intensity = compute_etas_intensity(eq_times, eq_locs, eq_mags, current_time)
    plot_map(eq_locs, stations, profile_names, lon_grid, lat_grid, intensity)
