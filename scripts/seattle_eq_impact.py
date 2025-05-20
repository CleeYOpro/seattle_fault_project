import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import OSM
from matplotlib.patches import Circle

seattle_eq = pd.read_csv('data/EQinSeattle.csv')

avg_lat = seattle_eq['latitude'].mean()
avg_lon = seattle_eq['longitude'].mean()

pred_mag = 7.0
pred_epicenter = (avg_lon, avg_lat)

impact_area = 10 ** (0.5 * pred_mag - 1.8)
impact_radius = np.sqrt(impact_area / np.pi)

print(f"Predicted epicenter: lat={avg_lat:.4f}, lon={avg_lon:.4f}")
print(f"Impact area: {impact_area:.2f} km^2, Impact radius: {impact_radius:.2f} km")

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Initialize geocoder
geolocator = Nominatim(user_agent="seattle_eq_impact")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

urm = pd.read_csv('data/Unreinforced_Masonry_Buildings_(URM).csv')

# Parse longitude/latitude directly from Long/Lat column
import re

def parse_point(point_str):
    # Handle case where input is already a float (from URM dataset)
    if isinstance(point_str, (float, int)):
        return point_str, point_str
    # Handle string POINT format
    if isinstance(point_str, str):
        match = re.search(r'POINT \(([-0-9.]+) ([-0-9.]+)\)', point_str)
        if match:
            # Return longitude first, then latitude (as per POINT format)
            return float(match.group(1)), float(match.group(2))
    return None, None

urm['longitude'], urm['latitude'] = zip(*urm['Long/Lat'].apply(parse_point))
urm = urm.dropna(subset=['latitude', 'longitude'])

MAP_EXTENT = [-122.45, -122.20, 47.48, 47.75]
fig = plt.figure(figsize=(10, 10))
tiler = OSM()
ax = plt.axes(projection=tiler.crs)
ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
ax.add_image(tiler, 12)

# Plot historical earthquakes
ax.scatter(
    seattle_eq['longitude'], seattle_eq['latitude'],
    s=30, c='blue', marker='o',
    transform=ccrs.PlateCarree(),
    label='Historical Earthquakes'
)

# Plot URM buildings using geocoded coordinates
ax.scatter(
    urm['longitude'], urm['latitude'],
    s=20, c='red', marker='o', alpha=0.1,
    transform=ccrs.PlateCarree(),
    label='URM Buildings'
)

ax.plot(
    avg_lon, avg_lat,
    marker='*', color='gold', markersize=18, markeredgecolor='black',
    transform=ccrs.PlateCarree(),
    label='Predicted M7.0 Epicenter'
)

deg_per_km = 1/111
radius_deg = impact_radius * deg_per_km
impact_circle = Circle(
    (avg_lon, avg_lat), radius_deg,
    edgecolor='blue', facecolor='none', linewidth=2, linestyle='--',
    transform=ccrs.PlateCarree(), label=f'Impact Radius ({impact_radius:.1f} km)'
)
ax.add_patch(impact_circle)

ax.legend(loc='upper right')
ax.set_title('Predicted M7.0 Earthquake Impact in Seattle with URM Buildings')

plt.show()
