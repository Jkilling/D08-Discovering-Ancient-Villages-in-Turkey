import os
import geopandas as gpd
from shapely.geometry import Point, Polygon
import numpy as np
import rasterio
import shutil
import random

datadir = os.getcwd()

# Data
tell_sites = gpd.read_file("data\\raw_data\\tell_sites.geojson").to_crs({'init': 'epsg:32637'})
poly = tell_sites.copy()
directory = r'data\\derived_data\\tiles'

# Buffer each point using a 900 meter circle radius
poly["geometry"] = poly.geometry.buffer(900)
poly_g = gpd.GeoSeries(poly['geometry'], crs='EPSG:32637')

findings_negative = []
confirmed_sites = os.listdir('data\\derived_data\\confirmed_sites')
confirmed_sites = ['data\\derived_data\\tiles\\' + x for x in confirmed_sites]

for filename in os.listdir(directory):
    raster = rasterio.open(directory+'\\'+filename, crs={'init': 'epsg:32637'})
    points = [Point(np.asarray(raster.bounds)[0], np.asarray(raster.bounds)[1]),
              Point(np.asarray(raster.bounds)[2], np.asarray(raster.bounds)[1]),
              Point(np.asarray(raster.bounds)[2], np.asarray(raster.bounds)[3]),
              Point(np.asarray(raster.bounds)[0], np.asarray(raster.bounds)[3])]

    bb = gpd.GeoSeries(Polygon(sum(map(list, (p.coords for p in points)), [])), crs='EPSG:32637')
    bools = poly_g.overlaps(bb.loc[0])

    if True in bools.values:
        print('Found overlapping polygon:', filename)
        findings_negative.append('data\\derived_data\\tiles\\'+filename)

    else:
        continue

for i in confirmed_sites:
    try:
        findings_negative.remove(i)
    except ValueError:
        pass

# Randomely chosen images
all_tiles = os.listdir('data\\derived_data\\tiles')
all_tiles = ['data\\derived_data\\tiles\\' + x for x in all_tiles]
used_ones = confirmed_sites+findings_negative

for i in used_ones:
    try:
        all_tiles.remove(i)
    except ValueError:
        pass

random_sites = random.sample(all_tiles, 1000-len(findings_negative))

# Copy found areas to external folder
for f in findings_negative+random_sites:
    shutil.copy(f, 'data\\derived_data\\negative_examples')

