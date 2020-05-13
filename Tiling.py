import os
import gdal

try:
    os.mkdir('data/derived_data')
    os.mkdir('data/derived_data/tiles')

except OSError:
    pass

data_folder = os.getcwd()+'\\data\\derived_data\\'
input_filename = 'Study_area_GT.tif'

out_path = data_folder+'tiles/'
output_filename = 'tile_'

tile_size_x = 70
tile_size_y = 90

ds = gdal.Open(data_folder + input_filename)
band = ds.GetRasterBand(1)
xsize = band.XSize
ysize = band.YSize

for i in range(0, xsize, tile_size_x):
    for j in range(0, ysize, tile_size_y):
        com_string = "gdal_translate -of GTIFF -srcwin " + str(i) + ", " + str(j) + ", " + str(
            tile_size_x) + ", " + str(tile_size_y) + " " + str(data_folder) + str(input_filename) + " " + str(
            out_path) + str(output_filename) + str(i) + "_" + str(j) + ".tif"
        os.system(com_string)

