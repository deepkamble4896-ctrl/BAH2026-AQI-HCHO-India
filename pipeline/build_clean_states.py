import json
from collections import defaultdict

with open('data/india_v2.geojson', encoding='utf-8') as f:
    districts = json.load(f)

print(f"Total district features: {len(districts['features'])}")

# Group district geometries by state name
state_groups = defaultdict(list)

for feat in districts['features']:
    props = feat['properties']
    state_name = props.get('st_nm')
    geom = feat.get('geometry')
    if state_name and geom:
        state_groups[state_name].append(geom)

print(f"States found: {len(state_groups)}")
for name in sorted(state_groups.keys()):
    print(f"  {name}: {len(state_groups[name])} district polygons")

# Merge all district polygons per state into one MultiPolygon
new_features = []
for state_name, geoms in state_groups.items():
    all_polys = []
    for geom in geoms:
        if geom['type'] == 'Polygon':
            all_polys.append(geom['coordinates'])
        elif geom['type'] == 'MultiPolygon':
            all_polys.extend(geom['coordinates'])

    new_features.append({
        'type': 'Feature',
        'properties': {'NAME_1': state_name},
        'geometry': {
            'type': 'MultiPolygon',
            'coordinates': all_polys
        }
    })

output = {
    'type': 'FeatureCollection',
    'features': new_features
}

with open('data/india_states_v2.geojson', 'w', encoding='utf-8') as f:
    json.dump(output, f)

print(f"\nSaved data/india_states_v2.geojson with {len(new_features)} states/UTs")