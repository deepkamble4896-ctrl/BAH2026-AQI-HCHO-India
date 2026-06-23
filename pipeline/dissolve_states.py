import json
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

with open('data/india_v2.geojson', encoding='utf-8') as f:
    districts = json.load(f)

print(f"Total district features: {len(districts['features'])}")

# Group district geometries by state
from collections import defaultdict
state_groups = defaultdict(list)

for feat in districts['features']:
    props = feat['properties']
    state_name = props.get('st_nm')
    geom = feat.get('geometry')
    if state_name and geom:
        state_groups[state_name].append(shape(geom))

print(f"States found: {len(state_groups)}")

new_features = []
for state_name, geoms in state_groups.items():
    # Dissolve all district polygons into one outer boundary
    merged = unary_union(geoms)
    new_features.append({
        'type': 'Feature',
        'properties': {'NAME_1': state_name},
        'geometry': mapping(merged)
    })
    print(f"  Dissolved {state_name}: {len(geoms)} districts -> 1 boundary")

output = {
    'type': 'FeatureCollection',
    'features': new_features
}

with open('data/india_states_dissolved.geojson', 'w', encoding='utf-8') as f:
    json.dump(output, f)

print(f"\nSaved data/india_states_dissolved.geojson with {len(new_features)} clean state boundaries")