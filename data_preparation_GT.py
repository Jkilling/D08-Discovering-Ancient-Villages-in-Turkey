import os
import gdal
import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import shutil

try:
    os.mkdir('data\\derived_data')
    os.mkdir('data\\derived_data\\tiles')
    os.mkdir('data\\derived_data\\confirmed_sites')

except OSError:
    pass

datadir = os.getcwd()

# Data
tell_sites = gpd.read_file("data\\raw_data\\tell_sites.geojson").to_crs({'init': 'epsg:32637'})

# Preparation for tiling
data_folder = os.getcwd() + '\\data'
input_filename = '\\raw_data\\study_area_hillshade_32637_GT.tif'

out_path = data_folder + '\\derived_data\\tiles\\'
output_filename = 'tile_'

# Define tiling size
gt_dem = rasterio.open(data_folder + input_filename, crs={'init': 'epsg:32637'})

tile_size_x = np.round(gt_dem.shape[0] / 55, 0).astype(int)  # in pixels not metrics!
tile_size_y = np.round(gt_dem.shape[1] / 55, 0).astype(int)

ds = gdal.Open(data_folder + input_filename)
band = ds.GetRasterBand(1)
xsize = band.XSize
ysize = band.YSize

# For loop for tiling using GDAL
for i in range(0, xsize, tile_size_x):
    for j in range(0, ysize, tile_size_y):
        com_string = "gdal_translate -of GTIFF -a_nodata 0 -srcwin " + str(i) + ", " + str(j) + ", " + str(
            tile_size_x) + ", " + str(tile_size_y) + " " + str(data_folder) + str(input_filename) + " " + str(
            out_path) + str(output_filename) + str(i) + "_" + str(j) + ".tif"
        os.system(com_string)

# Loop through directory
directory = r'data\\derived_data\\tiles'
findings = []
sites = gpd.GeoSeries(tell_sites['geometry'], crs='EPSG:32637')

for filename in os.listdir(directory):
    raster = rasterio.open(directory + '\\' + filename, crs={'init': 'epsg:32637'})
    points = [Point(np.asarray(raster.bounds)[0], np.asarray(raster.bounds)[1]),
              Point(np.asarray(raster.bounds)[2], np.asarray(raster.bounds)[1]),
              Point(np.asarray(raster.bounds)[2], np.asarray(raster.bounds)[3]),
              Point(np.asarray(raster.bounds)[0], np.asarray(raster.bounds)[3])]

    bb = gpd.GeoSeries(Polygon(sum(map(list, (p.coords for p in points)), [])), crs='EPSG:32637')
    bools = sites.within(bb.loc[0])

    if True in bools.values:
        print('Found containing polygon:', filename)
        findings.append('data\\derived_data\\tiles\\' + filename)

    else:
        continue

# Filter no data

for filename in os.listdir(directory):
    with rasterio.open(directory + '\\' + filename, crs={'EPSG:32637'}) as src:
        raster_array = src.read(1).ravel()

    if raster_array.any() == 0:
        print('Found NA:', filename)
        # findings.append('data\\derived_data\\tiles\\'+filename)

    else:
        continue

with rasterio.open('data\\derived_data\\tiles\\tile_2970_2214.tif', crs={'EPSG:32637'}) as src:
    raster_array = src.read(1).ravel()

# Copy found areas to external folder
for f in findings:
    shutil.copy(f, 'data\\derived_data\\confirmed_sites')