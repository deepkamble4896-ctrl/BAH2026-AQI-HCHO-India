import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
from matplotlib.path import Path as MplPath
import json

df = pd.read_csv('data/hcho_fire_grid.csv')

with open('data/india_states_dissolved.geojson', encoding='utf-8') as f:
    india_geojson = json.load(f)
print(f"States loaded: {len(india_geojson['features'])}")

def build_polys(geojson):
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
    return polys

india_polys = build_polys(india_geojson)

def build_mask(polys, lon_grid, lat_grid):
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    pts = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])
    inside = np.zeros(len(pts), dtype=bool)
    for poly in polys:
        inside |= poly.contains_points(pts)
    return inside.reshape(len(lat_grid), len(lon_grid))

def points_inside_india(lons, lats, polys):
    pts = np.column_stack([lons, lats])
    inside = np.zeros(len(pts), dtype=bool)
    for poly in polys:
        inside |= poly.contains_points(pts)
    return inside

lat_grid = np.linspace(6, 38, 450)
lon_grid = np.linspace(67, 98, 450)
mask = build_mask(india_polys, lon_grid, lat_grid)

seasons = [
    ('Stubble_Burning_2025', 'Punjab/Haryana Stubble Burning — Oct-Nov 2025'),
    ('Agricultural_Burning_2026', 'NE India/Myanmar Border Burning — Feb-Apr 2026'),
]

hcho_colorscale = [
    [0.00, '#0d1117'], [0.15, '#1a3a5c'], [0.35, '#2e7d8c'],
    [0.55, '#f4a637'], [0.75, '#f15a24'], [1.00, '#b71c1c'],
]

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=[s[1] for s in seasons],
    horizontal_spacing=0.08
)

for i, (season_key, title) in enumerate(seasons):
    s = df[df['season'] == season_key]
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    hcho_smooth = griddata(
        points=(s['lon'].values, s['lat'].values),
        values=s['HCHO'].values,
        xi=(lon_mesh, lat_mesh),
        method='cubic'
    )
    hcho_smooth_masked = np.where(mask, hcho_smooth, np.nan)

    fig.add_trace(go.Heatmap(
        z=hcho_smooth_masked, x=lon_grid, y=lat_grid,
        colorscale=hcho_colorscale, zmin=0, zmax=0.00035,
        zsmooth='best',
        showscale=(i==0),
        colorbar=dict(
            title=dict(text='HCHO<br>mol/m²', font=dict(color='white', size=11)),
            tickfont=dict(color='white', size=9), bgcolor='rgba(20,20,20,0.8)',
            x=0.46 if i==0 else 1.02, len=0.8
        ) if i==0 else None,
        hovertemplate='Lat: %{y:.1f}°N<br>Lon: %{x:.1f}°E<br>HCHO: %{z:.2e}<extra></extra>'
    ), row=1, col=i+1)

    # Fire markers - significant clusters, INSIDE India only
    fires = s[s['fire_count'] > 200].copy()
    fires['inside'] = points_inside_india(fires['lon'].values, fires['lat'].values, india_polys)
    fires = fires[fires['inside']]
    fires['size'] = np.clip(np.sqrt(fires['fire_count'])/3, 4, 22)

    fig.add_trace(go.Scatter(
        x=fires['lon'], y=fires['lat'], mode='markers',
        marker=dict(
            size=fires['size'],
            color='#ff3b30', opacity=0.55,
            line=dict(color='#ffcc00', width=1)
        ),
        showlegend=False,
        hovertemplate='Fire cluster<extra></extra>'
    ), row=1, col=i+1)

    # State boundaries
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
                line=dict(color='rgba(255,255,255,0.4)', width=0.7),
                showlegend=False, hoverinfo='skip'
            ), row=1, col=i+1)

    fig.update_xaxes(range=[67, 98], showgrid=False, color='white',
                     constrain='domain', row=1, col=i+1)
    fig.update_yaxes(range=[6, 38], showgrid=False, color='white',
                     scaleanchor=f'x{i+1}', scaleratio=1, row=1, col=i+1)

fig.update_layout(
    title=dict(
        text='<b>HCHO Hotspot Map — India Biomass Burning Seasons</b><br>'
             '<sup>Sentinel-5P OFFL + NASA FIRMS · Red dots = active fire clusters (India only)</sup>',
        font=dict(color='white', size=18), x=0.5, xanchor='center'
    ),
    paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
    width=1500, height=750,
    font=dict(color='white'),
    margin=dict(l=50, r=100, t=100, b=50)
)
for ann in fig['layout']['annotations']:
    ann['font'] = dict(color='white', size=14)

fig.write_html('hcho_hotspot_map.html')
print("Saved hcho_hotspot_map.html")

try:
    fig.write_image('hcho_hotspot_map.png', width=2000, height=1000, scale=2)
    print("Saved hcho_hotspot_map.png")
except Exception as e:
    print(f"PNG failed: {e}")