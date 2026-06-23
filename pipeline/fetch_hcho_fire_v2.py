import ee
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

ee.Initialize(project='aqi-hcho-isro')

seasons = [
    ('Stubble_Burning_2025', '2025-10-15', '2025-11-30'),
    ('Agricultural_Burning_2026', '2026-02-01', '2026-04-30'),
]

india = ee.Geometry.Rectangle([68, 8, 97, 37])

# Build 0.5-degree grid cells as polygons (not points) for fire counting
grid_lats = np.round(np.arange(8.5, 37.0, 0.5), 2)
grid_lons = np.round(np.arange(68.5, 97.0, 0.5), 2)

cells = []
for lat in grid_lats:
    for lon in grid_lons:
        cell = ee.Feature(
            ee.Geometry.Rectangle([lon-0.25, lat-0.25, lon+0.25, lat+0.25]),
            {'lat': float(lat), 'lon': float(lon)}
        )
        cells.append(cell)
cell_fc = ee.FeatureCollection(cells)
print(f"Grid cells: {len(cells)}")

all_results = []

for season_name, start, end in seasons:
    print(f"\n=== {season_name}: {start} to {end} ===")

    # HCHO smooth column density
    hcho_img = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_HCHO') \
        .filterDate(start, end) \
        .select('tropospheric_HCHO_column_number_density') \
        .mean()

    print("Sampling HCHO at grid points (buffered)...")
    buffered_points = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(lon), float(lat)]).buffer(15000), {'lat': float(lat), 'lon': float(lon)})
        for lat in grid_lats for lon in grid_lons
    ])
    hcho_result = hcho_img.reduceRegions(
        collection=buffered_points,
        reducer=ee.Reducer.mean(),
        scale=10000
    )
    hcho_data = hcho_result.getInfo()
    hcho_dict = {}
    for feat in hcho_data['features']:
        props = feat['properties']
        key = (round(props['lat'], 2), round(props['lon'], 2))
        hcho_dict[key] = props.get('mean') or 0
    print(f"  HCHO points: {len(hcho_dict)}")

    # Fire count per grid cell using reduceRegions (counts pixels inside each polygon)
    print("Counting fires per grid cell...")
    fire_img = ee.ImageCollection('FIRMS').filterDate(start, end).select('T21').count()

    fire_counts = fire_img.reduceRegions(
        collection=cell_fc,
        reducer=ee.Reducer.count(),
        scale=1000
    )
    fire_data = fire_counts.getInfo()
    fire_dict = {}
    for feat in fire_data['features']:
        props = feat['properties']
        key = (round(props['lat'], 2), round(props['lon'], 2))
        fire_dict[key] = props.get('count') or 0
    print(f"  Fire-count cells: {len(fire_dict)}")

    rows = []
    for lat in grid_lats:
        for lon in grid_lons:
            key = (round(lat, 2), round(lon, 2))
            rows.append({
                'lat': lat, 'lon': lon,
                'HCHO': hcho_dict.get(key, 0),
                'fire_count': fire_dict.get(key, 0),
                'season': season_name
            })

    df = pd.DataFrame(rows)
    print(f"  HCHO mean: {df['HCHO'].mean():.6f} | Cells with fires: {(df['fire_count']>0).sum()}")
    all_results.append(df)

final = pd.concat(all_results, ignore_index=True)
final.to_csv('data/hcho_fire_grid.csv', index=False)
print(f"\nSaved {len(final)} total rows to data/hcho_fire_grid.csv")

for season_name, _, _ in seasons:
    s = final[final['season'] == season_name]
    print(f"\n{season_name}:")
    print(f"  HCHO range: {s['HCHO'].min():.6f} to {s['HCHO'].max():.6f}")
    print(f"  Fire cells: {(s['fire_count']>0).sum()} / {len(s)}")
    top = s.nlargest(5, 'fire_count')[['lat','lon','fire_count','HCHO']]
    print(top)