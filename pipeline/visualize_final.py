import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
import json

df = pd.read_csv('data/physics_surface_grid_recent.csv')
print(f"Grid points: {len(df)}")

# Compute AQI using official NAQI sub-index breakpoints
breakpoints = {
    'PM2.5': [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400),(251,500,401,500)],
    'NO2':   [(0,40,0,50),(41,80,51,100),(81,180,101,200),(181,280,201,300),(281,400,301,400),(401,500,401,500)],
    'SO2':   [(0,40,0,50),(41,80,51,100),(81,380,101,200),(381,800,201,300),(801,1600,301,400),(1601,2000,401,500)],
    'CO':    [(0,1,0,50),(1.1,2,51,100),(2.1,10,101,200),(10.1,17,201,300),(17.1,34,301,400),(34.1,50,401,500)],
    'O3':    [(0,50,0,50),(51,100,51,100),(101,168,101,200),(169,208,201,300),(209,748,301,400),(749,900,401,500)],
}

def sub_index(conc, pollutant):
    if pd.isna(conc) or pollutant not in breakpoints:
        return np.nan
    for c_lo, c_hi, i_lo, i_hi in breakpoints[pollutant]:
        if c_lo <= conc <= c_hi:
            return i_lo + (i_hi - i_lo) * (conc - c_lo) / (c_hi - c_lo)
    c_lo, c_hi, i_lo, i_hi = breakpoints[pollutant][-1]
    if conc > c_hi:
        return min(500, i_hi + (conc - c_hi))
    return np.nan

df['PM2.5_sub'] = df['PM2.5'].apply(lambda x: sub_index(x, 'PM2.5'))
df['NO2_sub']   = df['NO2'].apply(lambda x: sub_index(x, 'NO2'))
df['SO2_sub']   = df['SO2'].apply(lambda x: sub_index(x, 'SO2'))
df['CO_sub']    = df['CO'].apply(lambda x: sub_index(x, 'CO'))
df['O3_sub']    = df['O3'].apply(lambda x: sub_index(x, 'O3'))

sub_cols = ['PM2.5_sub','NO2_sub','SO2_sub','CO_sub','O3_sub']
df['AQI'] = df[sub_cols].max(axis=1)

print("AQI distribution:")
print(df['AQI'].describe())

with open('data/india_states_dissolved.geojson', encoding='utf-8') as f:
    india_geojson = json.load(f)
print(f"States loaded: {len(india_geojson['features'])}")

lat_grid = np.linspace(6, 38, 350)
lon_grid = np.linspace(67, 98, 350)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

aqi_smooth = griddata(
    points=(df['lon'].values, df['lat'].values),
    values=df['AQI'].values,
    xi=(lon_mesh, lat_mesh),
    method='cubic'
)

from matplotlib.path import Path as MplPath

def build_mask(geojson, lon_grid, lat_grid):
    polys = []
    for feature in geojson['features']:
        geom = feature['geometry']
        coords_list = []
        if geom['type'] == 'Polygon':
            coords_list = geom['coordinates']
        elif geom['type'] == 'MultiPolygon':
            for p in geom['coordinates']:
                coords_list.extend(p)
        for ring in coords_list:
            polys.append(MplPath(np.array(ring)))

    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    pts = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])
    inside = np.zeros(len(pts), dtype=bool)
    for poly in polys:
        inside |= poly.contains_points(pts)
    return inside.reshape(len(lat_grid), len(lon_grid))

print("Building India mask...")
mask = build_mask(india_geojson, lon_grid, lat_grid)
aqi_smooth_masked = np.where(mask, aqi_smooth, np.nan)

aqi_colorscale = [
    [0.00, '#00e400'], [0.10, '#a3ff00'], [0.20, '#ffff00'],
    [0.40, '#ff7e00'], [0.60, '#ff0000'], [0.80, '#8f3f97'], [1.00, '#7e0023'],
]

fig = go.Figure()

fig.add_trace(go.Heatmap(
    z=aqi_smooth_masked, x=lon_grid, y=lat_grid,
    colorscale=aqi_colorscale, zmin=0, zmax=500, opacity=0.9, showscale=True,
    colorbar=dict(
        title=dict(text='AQI', font=dict(color='white', size=14)),
        tickvals=[0, 50, 100, 200, 300, 400, 500],
        ticktext=['0', '50 Good', '100 Sat.', '200 Mod.', '300 Poor', '400 V.Poor', '500 Severe'],
        tickfont=dict(color='white', size=11), bgcolor='rgba(20,20,20,0.8)',
        bordercolor='rgba(255,255,255,0.2)', borderwidth=1, len=0.85, x=1.02
    ),
    hovertemplate='Lat: %{y:.1f}°N<br>Lon: %{x:.1f}°E<br>AQI: %{z:.0f}<extra></extra>'
))

for feature in india_geojson['features']:
    geom = feature['geometry']
    coords_list = []
    if geom['type'] == 'Polygon':
        coords_list = [geom['coordinates'][0]]
    elif geom['type'] == 'MultiPolygon':
        coords_list = [p[0] for p in geom['coordinates']]
    for coords in coords_list:
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        fig.add_trace(go.Scatter(
            x=lons, y=lats, mode='lines',
            line=dict(color='rgba(255,255,255,0.6)', width=1),
            showlegend=False, hoverinfo='skip'
        ))

city_coords = {
    'Delhi': (28.6, 77.2), 'Mumbai': (19.0, 72.8), 'Kolkata': (22.5, 88.3),
    'Chennai': (13.0, 80.2), 'Bengaluru': (12.9, 77.6), 'Hyderabad': (17.4, 78.5),
    'Pune': (18.5, 73.9), 'Ahmedabad': (23.0, 72.6), 'Lucknow': (26.8, 80.9),
    'Patna': (25.6, 85.1), 'Jaipur': (26.9, 75.8), 'Bhopal': (23.2, 77.4),
    'Nagpur': (21.1, 79.1), 'Surat': (21.2, 72.8), 'Kanpur': (26.4, 80.3),
    'Visakhapatnam': (17.7, 83.3), 'Indore': (22.7, 75.8), 'Coimbatore': (11.0, 76.9),
    'Kochi': (9.9, 76.2), 'Guwahati': (26.1, 91.7), 'Bhubaneswar': (20.3, 85.8),
    'Chandigarh': (30.7, 76.7), 'Amritsar': (31.6, 74.9), 'Varanasi': (25.3, 83.0),
    'Agra': (27.2, 78.0), 'Jodhpur': (26.3, 73.0), 'Raipur': (21.2, 81.6),
    'Ranchi': (23.3, 85.3), 'Thiruvananthapuram': (8.5, 76.9), 'Madurai': (9.9, 78.1),
    'Srinagar': (34.1, 74.8), 'Leh': (34.2, 77.6), 'Shillong': (25.6, 91.9),
    'Imphal': (24.8, 93.9), 'Gangtok': (27.3, 88.6), 'Panaji': (15.5, 73.8),
    'Port Blair': (11.7, 92.7), 'Dehradun': (30.3, 78.0), 'Shimla': (31.1, 77.2),
}

fig.add_trace(go.Scatter(
    x=[v[1] for v in city_coords.values()], y=[v[0] for v in city_coords.values()],
    mode='markers+text', marker=dict(size=5, color='white', line=dict(color='black', width=1)),
    text=list(city_coords.keys()), textposition='top center',
    textfont=dict(color='white', size=8, family='Arial'),
    showlegend=False, hovertemplate='<b>%{text}</b><extra></extra>'
))

fig.update_layout(
    title=dict(
        text='<b>India Surface AQI Map — June 2026</b><br>'
             '<sup>Sentinel-5P + MODIS + ERA5 · Physics-Based Conversion · Validated against CPCB</sup>',
        font=dict(color='white', size=20), x=0.5, xanchor='center', y=0.97
    ),
    paper_bgcolor='#0d1117', plot_bgcolor='#0d1117', width=1100, height=950,
    xaxis=dict(title='Longitude', color='white', range=[67, 98], showgrid=False, constrain='domain'),
    yaxis=dict(title='Latitude', color='white', range=[6, 38], showgrid=False, scaleanchor='x', scaleratio=1),
    margin=dict(l=60, r=120, t=80, b=60), font=dict(color='white')
)

fig.write_html('aqi_map_final.html')
print("Saved aqi_map_final.html")
try:
    fig.write_image('aqi_map_final.png', width=1600, height=1300, scale=2)
    print("Saved aqi_map_final.png")
except Exception as e:
    print(f"PNG failed: {e}")