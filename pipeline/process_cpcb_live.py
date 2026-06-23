import pandas as pd
import numpy as np

# Load raw long-format live data (downloaded from data.gov.in)
df = pd.read_csv('data/cpcb_live_raw.csv')

df['pollutant_avg'] = pd.to_numeric(df['pollutant_avg'], errors='coerce')
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

# Pivot to wide format: one row per station, pollutants as columns
pivot = df.pivot_table(
    index=['station', 'city', 'state', 'latitude', 'longitude'],
    columns='pollutant_id',
    values='pollutant_avg',
    aggfunc='mean'
).reset_index()

# Official CPCB NAQI breakpoints
breakpoints = {
    'PM2.5': [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400),(251,500,401,500)],
    'PM10':  [(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),(351,430,301,400),(431,600,401,500)],
    'NO2':   [(0,40,0,50),(41,80,51,100),(81,180,101,200),(181,280,201,300),(281,400,301,400),(401,500,401,500)],
    'SO2':   [(0,40,0,50),(41,80,51,100),(81,380,101,200),(381,800,201,300),(801,1600,301,400),(1601,2000,401,500)],
    'CO':    [(0,1,0,50),(1.1,2,51,100),(2.1,10,101,200),(10.1,17,201,300),(17.1,34,301,400),(34.1,50,401,500)],  # mg/m3
    'OZONE': [(0,50,0,50),(51,100,51,100),(101,168,101,200),(169,208,201,300),(209,748,301,400),(749,900,401,500)],
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

# IMPORTANT: CPCB live API reports CO in µg/m³, but breakpoints need mg/m³
pivot['CO_mgm3'] = pivot['CO'] / 1000.0

pivot['PM2.5_subindex'] = pivot['PM2.5'].apply(lambda x: sub_index(x, 'PM2.5'))
pivot['PM10_subindex']  = pivot['PM10'].apply(lambda x: sub_index(x, 'PM10'))
pivot['NO2_subindex']   = pivot['NO2'].apply(lambda x: sub_index(x, 'NO2'))
pivot['SO2_subindex']   = pivot['SO2'].apply(lambda x: sub_index(x, 'SO2'))
pivot['CO_subindex']    = pivot['CO_mgm3'].apply(lambda x: sub_index(x, 'CO'))
pivot['OZONE_subindex'] = pivot['OZONE'].apply(lambda x: sub_index(x, 'OZONE'))

subindex_cols = ['PM2.5_subindex','PM10_subindex','NO2_subindex','SO2_subindex','CO_subindex','OZONE_subindex']
pivot['AQI_live'] = pivot[subindex_cols].max(axis=1)

# Keep clean output, rename OZONE -> O3 for consistency with your existing pipeline
final = pivot[['station','city','state','latitude','longitude','PM2.5','PM10','NO2','SO2','CO','OZONE','AQI_live']]
final = final.rename(columns={'OZONE': 'O3'})

final.to_csv('data/cpcb_live_validation.csv', index=False)
print(f"Saved {len(final)} stations to data/cpcb_live_validation.csv")
print(f"\nAQI distribution:")
print(final['AQI_live'].describe())