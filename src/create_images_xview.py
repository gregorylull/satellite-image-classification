
# conda activate solaris
# (glull) follow the solaris installation and tutorial to create
# spacenet masks https://solaris.readthedocs.io/en/latest/tutorials/notebooks/api_masks_tutorial.html
# note, can also create multiple layers if necessary (i.e. if there's other data like roads).

import solaris as sol
from solaris.data import data_dir
import os
import skimage
import geopandas as gpd
from matplotlib import pyplot as plt
from shapely.ops import cascaded_union

from tqdm import tqdm
import pandas as pd
import numpy as np
import glob

# arguments to mask creation
# df: A pandas.DataFrame or geopandas.GeoDataFrame containing polygons in one column.
# out_file: An optional argument to specify a filepath to save outputs to.
# reference_im: A georegistered image covering the same geography as df. This is optional, but if it’s not provided and you wish to convert the polygons from a geospatial CRS to pixel coordinates, you must provide affine_obj.
# geom_col: An optional argument specifying which column holds polygons in df. This defaults to "geometry", the default geometry column in GeoDataFrames.
# do_transform: Should polygons be transformed from a geospatial CRS to pixel coordinates? solaris will try to infer whether or not this is necessary, but you can force behavior by providing True or False with this argument.
# affine_obj: An affine object specifying a transform to convert polygons from a geospatial CRS to pixel coordinates. This optional (unless do_transform=True and you don’t provide reference_im), and is superceded by reference_im if that argument is also provided.
# shape: Optionally, the (X, Y) bounds of the output image. If reference_im is provided, the shape of that image supercedes this argument.
# out_type: Either int or float, the dtype of the output. Defaults to int.
# burn_value: The value that pixels falling within the polygon will be set to. Defaults to 255.
# burn_field: Optionally, a column within df that specifies the value to set for each individual polygon. This can be used if you have multiple classes.

# this is mask creation
data = 'data/spacenet/'
AOI = 'AOI_2_Vegas_Train/'
image_prefix = 'RGB-PanSharpen_'
geojson_prefix = 'buildings_'

image_dir = os.path.join(data, AOI, 'RGB-PanSharpen')
# image_dir = os.path.join(data, AOI, 'test_RGB-PanSharpen')

geojson_dir = os.path.join(data, AOI, 'geojson', 'buildings')
output_dir = os.path.join(data, AOI, 'RGB-PanSharpen_masks')

image_id = 'AOI_2_Vegas_img1'

# # create a single mask
# fp_mask = sol.vector.mask.footprint_mask(
#     df = os.path.join(geojson_dir, geojson_filename),
#     reference_im = os.path.join(image_dir, image_filename),
#     out_file = os.path.join(output_dir, output_filename)
# )


def create_mask(image_id):

    image_filename = f'{image_prefix}{image_id}.tif'
    geojson_filename = f'{geojson_prefix}{image_id}.geojson'
    output_filename = f'{image_id}_mask.tif'

    df = os.path.join(geojson_dir, geojson_filename)
    reference_im = os.path.join(image_dir, image_filename)
    out_file = os.path.join(output_dir, output_filename)

    fp_mask = sol.vector.mask.footprint_mask(
        df=df,
        reference_im=reference_im,
        out_file=out_file
    )


def create_masks(image_ids):
    for index, image_id in tqdm(enumerate(image_ids)):
        create_mask(image_id)


def get_image_ids():

    summary_csv = os.path.join(
        data, AOI, 'summaryData', 'AOI_2_Vegas_Train_Building_Solutions.csv'
    )
    df = pd.read_csv(summary_csv)
    imageids = df['ImageId'].unique()

    return sorted(imageids)


def main():

    print('Getting ids...')
    image_ids = get_image_ids()
    print(f'  finished retrieving image_ids {len(image_ids)}')

    print('Creating masks...')
    create_masks(image_ids)
    print('  finished creating masks')


if __name__ == '__main__':
    print('\nRunning mask program\n')
    main()

# this is a preview of the images
# image = skimage.io.imread(os.path.join(data, 'sample_geotiff.tif'))
# f, axarr = plt.subplots(figsize=(10, 10))

# plt.savefig(f'{folder}example_fig.png')
# plt.close()

# plt.imshow(fp_mask, cmap='gray')
# f, ax = plt.subplots(1, 2, figsize=(20, 10))

# plt.savefig(os.path.join(data, f'image_mask_{image_id}.png'))
