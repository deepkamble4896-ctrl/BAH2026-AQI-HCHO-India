import ee
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

ee.Initialize(project='aqi-hcho-isro')

start = '2026-03-01'
end   = '2026-03-31'
print(f"Fetching recent satellite + ERA5 data: {start} to {end}")

grid_lats = np.round(np.arange(8.5, 37.0, 0.5), 2)
grid_lons = np.round(np.arange(68.5, 97.0, 0.5), 2)
features = [
    ee.Feature(ee.Geometry.Point([float(lon), float(lat)]), {'lat': float(lat), 'lon': float(lon)})
    for lat in grid_lats for lon in grid_lons
]
points = ee.FeatureCollection(features)
print(f"Grid points: {len(grid_lats) * len(grid_lons)}")

sat = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2') \
    .filterDate(start, end).select('tropospheric_NO2_column_number_density').mean().rename('NO2_col') \
    .addBands(ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_SO2')
        .filterDate(start, end).select('SO2_column_number_density').mean().rename('SO2_col')) \
    .addBands(ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_CO')
        .filterDate(start, end).select('CO_column_number_density').mean().rename('CO_col')) \
    .addBands(ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_O3')
        .filterDate(start, end).select('O3_column_number_density').mean().rename('O3_col')) \
    .addBands(ee.ImageCollection('MODIS/061/MCD19A2_GRANULES')
        .filterDate(start, end).select('Optical_Depth_047').mean().rename('AOD')) \
    .addBands(ee.ImageCollection('ECMWF/ERA5/HOURLY')
        .filterDate(start, end).select('boundary_layer_height').mean().rename('BLH')) \
    .addBands(ee.ImageCollection('ECMWF/ERA5/HOURLY')
        .filterDate(start, end).select('u_component_of_wind_10m').mean().rename('u_wind')) \
    .addBands(ee.ImageCollection('ECMWF/ERA5/HOURLY')
        .filterDate(start, end).select('v_component_of_wind_10m').mean().rename('v_wind'))

print("Sampling grid (1-2 min)...")
sampled = sat.sampleRegions(collection=points, scale=10000, geometries=True)
data = sampled.getInfo()
print(f"Got {len(data['features'])} points")

rows = []
for feat in data['features']:
    props = feat['properties']
    coords = feat['geometry']['coordinates']
    u = props.get('u_wind') or 0
    v = props.get('v_wind') or 0
    rows.append({
        'lat': coords[1], 'lon': coords[0],
        'NO2_col': props.get('NO2_col') or 0,
        'SO2_col': props.get('SO2_col') or 0,
        'CO_col':  props.get('CO_col')  or 0,
        'O3_col':  props.get('O3_col')  or 0,
        'AOD':     props.get('AOD') or 0,
        'BLH':     props.get('BLH') or 500,
        'wind_speed': (u**2 + v**2) ** 0.5,
    })

df = pd.DataFrame(rows)
print(f"Valid points after sampling: {len(df)}")

MOLAR_MASS = {'NO2': 46.01, 'SO2': 64.07}
df['NO2_col'] = df['NO2_col'].clip(lower=0)
df['SO2_col'] = df['SO2_col'].clip(lower=0)
df['NO2'] = (df['NO2_col'] * MOLAR_MASS['NO2'] * 1e6) / df['BLH']
df['SO2'] = (df['SO2_col'] * MOLAR_MASS['SO2'] * 1e6) / df['BLH']
df['CO'] = 0.3 + 1.2 * (df['CO_col'] - df['CO_col'].min()) / (df['CO_col'].max() - df['CO_col'].min())
df['O3'] = 15 + 70 * (df['O3_col'] - df['O3_col'].min()) / (df['O3_col'].max() - df['O3_col'].min())

# PM2.5 from AOD - BLH-based physics, with BLH floored to avoid
# runaway amplification in humid/monsoon regions where low BLH
# reflects moisture, not pollution trapping
SCALE_HEIGHT = 8000
AOD_CEILING = 1200
BLH_FLOOR = 400

df['AOD_clean'] = df['AOD'].clip(upper=AOD_CEILING)
df['AOD_norm'] = df['AOD_clean'] / 1000
df['BLH_for_pm25'] = df['BLH'].clip(lower=BLH_FLOOR)

df['PM2.5'] = (df['AOD_norm'] * SCALE_HEIGHT) / df['BLH_for_pm25'] * 12.2

n_flagged = (df['AOD'] > AOD_CEILING).sum()
n_floored = (df['BLH'] < BLH_FLOOR).sum()
print(f"\nFlagged {n_flagged} grid points ({n_flagged/len(df)*100:.1f}%) with likely cloud-contaminated AOD")
print(f"Floored {n_floored} grid points ({n_floored/len(df)*100:.1f}%) with BLH below {BLH_FLOOR}m")

print("\n=== Recent-window (Jun 6-20) surface concentrations ===")
print(df[['NO2','SO2','CO','O3','PM2.5']].describe())

# Quick regional sanity check
delhi = df[(df['lat']>27.5)&(df['lat']<29.5)&(df['lon']>76.5)&(df['lon']<78)]
ne = df[(df['lat']>22)&(df['lat']<28)&(df['lon']>88)&(df['lon']<96)]
print(f"\nDelhi PM2.5 mean: {delhi['PM2.5'].mean():.1f}")
print(f"Northeast PM2.5 mean: {ne['PM2.5'].mean():.1f}")

df.to_csv('data/physics_surface_grid_recent.csv', index=False)
print("\nSaved data/physics_surface_grid_recent.csv")