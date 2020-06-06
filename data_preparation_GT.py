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
data_folder = os.getcwd()+'\\data'
input_filename = '\\raw_data\\study_area_hillshade_32637_GT.tif'

out_path = data_folder+'\\derived_data\\tiles\\'
output_filename = 'tile_'

# Define tiling size
gt_dem = rasterio.open(data_folder+input_filename, crs={'init': 'epsg:32637'})

tile_size_x = np.round(gt_dem.shape[0]/99, 0).astype(int)  # in pixels not metrics!
tile_size_y = np.round(gt_dem.shape[1]/119, 0).astype(int)

ds = gdal.Open(data_folder + input_filename)
band = ds.GetRasterBand(1)
xsize = band.XSize
ysize = band.YSize

# For loop for tiling using GDAL
for i in range(0, xsize, tile_size_x):
    for j in range(0, ysize, tile_size_y):
        com_string = "gdal_translate -of GTIFF -srcwin " + str(i) + ", " + str(j) + ", " + str(
            tile_size_x) + ", " + str(tile_size_y) + " " + str(data_folder) + str(input_filename) + " " + str(
            out_path) + str(output_filename) + str(i) + "_" + str(j) + ".tif"
        os.system(com_string)


# Filter no data
directory = r'data\\derived_data\\tiles'
nodatas = []
for filename in os.listdir(directory):
    with rasterio.open(directory + '\\' + filename, crs={'EPSG:32637'}) as src:
        raster_array = src.read(1).ravel()

    if (raster_array == 0).any():
        print('Found NA:', filename)
        nodatas.append('data\\derived_data\\tiles\\' + filename)

    else:
        continue


for f in nodatas:
   os.remove(f)

# Loop through directory
directory = r'data\\derived_data\\tiles'
findings = []
sites = gpd.GeoSeries(tell_sites['geometry'], crs='EPSG:32637')

for filename in os.listdir(directory):
    raster = rasterio.open(directory+'\\'+filename, crs={'init': 'epsg:32637'})
    points = [Point(np.asarray(raster.bounds)[0], np.asarray(raster.bounds)[1]),
              Point(np.asarray(raster.bounds)[2], np.asarray(raster.bounds)[1]),
              Point(np.asarray(raster.bounds)[2], np.asarray(raster.bounds)[3]),
              Point(np.asarray(raster.bounds)[0], np.asarray(raster.bounds)[3])]

    bb = gpd.GeoSeries(Polygon(sum(map(list, (p.coords for p in points)), [])), crs='EPSG:32637')
    bools = sites.within(bb.loc[0])

    if True in bools.values:
        print('Found containing polygon:', filename)
        findings.append('data\\derived_data\\tiles\\'+filename)

    else:
        continue


# Copy found areas to external folder
for f in findings:
    shutil.copy(f, 'data\\derived_data\\confirmed_sites')

# Buffer each point using a 900 meter circle radius
# Buffer each point using a 900 meter circle radius
datadir = os.getcwd()
tell_sites = gpd.read_file("data/raw_data/tell_sites.geojson").to_crs({'init': 'epsg:32637'})
poly = tell_sites.copy()
directory = r'data/derived_data/tiles'
try:
    os.mkdir('data/derived_data/negative_examples')

except OSError:
    pass

poly['geometry'] = poly.geometry.buffer(900)
poly_g = gpd.GeoSeries(poly['geometry'], crs='EPSG:32637')

findings_negative = []
confirmed_sites = os.listdir('data/derived_data/confirmed_sites')
eliminated_sites = ['tile_500_1425.tif', 'tile_2200_50.tif', 'tile_2275_375.tif',
                    'tile_2000_1850.tif', 'tile_1850_1950.tif']

for i in eliminated_sites:
    try:
        confirmed_sites.remove(i)
    except ValueError:
        pass

confirmed_sites = ['data/derived_data/tiles/' + x for x in confirmed_sites]

for filename in os.listdir(directory):
    raster = rasterio.open(directory + '/' + filename, crs={'init': 'epsg:32637'})
    points = [Point(np.asarray(raster.bounds)[0], np.asarray(raster.bounds)[1]),
              Point(np.asarray(raster.bounds)[2], np.asarray(raster.bounds)[1]),
              Point(np.asarray(raster.bounds)[2], np.asarray(raster.bounds)[3]),
              Point(np.asarray(raster.bounds)[0], np.asarray(raster.bounds)[3])]

    bb = gpd.GeoSeries(Polygon(sum(map(list, (p.coords for p in points)), [])), crs='EPSG:32637')
    bools = poly_g.overlaps(bb.loc[0])

    if True in bools.values:
        # print('Found overlapping polygon:', filename)
        findings_negative.append('data/derived_data/tiles/' + filename)

    else:
        continue

    for i in confirmed_sites:
        try:
            findings_negative.remove(i)
        except ValueError:
            pass

for f in findings_negative:
    shutil.copy(f, 'data/derived_data/negative_examples')
print(len(os.listdir('data/derived_data/negative_examples')))
