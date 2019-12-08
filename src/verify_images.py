# (glull) this file will verify split images can be restiched back to the original.
# Stich together an image

import os
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm_notebook, tnrange, tqdm
from skimage.external import tifffile
from matplotlib import pyplot as plt

from src.utilities import data as gldata

SAVE_FIG = True

data = 'data/spacenet/'
AOI = 'AOI_2_Vegas_Train/'
image_prefix = 'RGB-PanSharpen_'
geojson_prefix = 'buildings_'
mask_postfix = '_mask'
assets_dir = os.path.join('assets', 'verify_images/')

IMAGE_DIR = os.path.join(data, AOI, 'RGB-PanSharpen')
GEOJSON_DIR = os.path.join(data, AOI, 'geojson', 'buildings')
OUTPUT_DIR = os.path.join(data, AOI, 'RGB-PanSharpen_masks')

vegas_csv = 'AOI_2_Vegas_Train_Building_Solutions.csv'

split_dimension = 128
image_dimension = 640
image_channels = 3
mask_channels = 1

# train_percentage = 1
train_percentage = 0.001

ids = gldata.get_image_ids()

if train_percentage != 1:
    train_index = int((len(ids) * train_percentage) // 1)
    ids = ids[:train_index]

print(f'Splitting {len(ids)} images')


def get_image_path(image_id):
    image = f'data/spacenet/AOI_2_Vegas_Train/RGB-PanSharpen_resize640_split128/RGB-PanSharpen_AOI_2_Vegas_img{image_id}_part*'
    mask = f'data/spacenet/AOI_2_Vegas_Train/RGB-PanSharpen_masks_resize640_split128/AOI_2_Vegas_img{image_id}_mask_part*'

    return image, mask


# 12 30 42image_prefix
image_ids = [1, 3, 4, 7, 8, 9, 10, 12, 13, 14, 18, 30, 32, 36, 37, 40]
image_short_ids = [3, 4]
image_id = [12]
single_image, single_mask = get_image_path(image_id)
parameters = ''

RGB_bits = 2047  # RGB images
mask_bits = 255  # grayscale


def stich_single(image_filepath, mask_filepath):
    image_paths = glob.glob(image_filepath)
    mask_paths = glob.glob(mask_filepath)

    image_paths = sorted(image_paths)
    mask_paths = sorted(mask_paths)

    img = tifffile.imread(image_paths[8])
    mask = tifffile.imread(mask_paths[8])

    print(img.max(), mask.max())

    all_images = []
    for image_path in image_paths:
        all_images.append(tifffile.imread(image_path) / RGB_bits)

    all_masks = []
    for mask_path in mask_paths:
        all_masks.append(tifffile.imread(mask_path) / mask_bits)

    stiched = gldata.stich_images(all_images)

    stiched_masks = gldata.stich_images(all_masks)

    return stiched, stiched_masks


def savefig(fig, name, save=SAVE_FIG):
    if save:
        fig.savefig(
            name,
            bbox_inches='tight'
        )


def save_images(id, x_image, y_image, parameters=''):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    ax[0].imshow(x_image, interpolation='bilinear')
    ax[0].set_title(f'Satellite {id}')

    ax[1].imshow(y_image.squeeze(), interpolation='bilinear', cmap='gray')
    ax[1].set_title(f'Buildings {id}')

    fig_name = os.path.join(
        assets_dir, f'{id}_mask_comparison_{parameters}.png')
    print(f'\nSaving figure {fig_name}\n')
    savefig(fig, fig_name)


def save_single_multiple(image_ids):
    for index, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):
        single_image_path, single_mask_path = get_image_path(image_id)
        img, mask = stich_single(single_image_path, single_mask_path)
        save_images(image_id, img, mask, 'single')


source = image_ids

save_single_multiple(source)


def save_with_get_data(image_ids):
    ids = [f'AOI_2_Vegas_img{image_id}' for image_id in image_ids]
    X, y = gldata.get_data(
        ids,
        split_dimension,
        image_dimension,
        image_channels,
        mask_channels,
    )

    for index, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):
        start = index * 25
        end = start + 25
        x_img = X[start:end]
        y_img = y[start:end]
        img = gldata.stich_images(x_img)
        mask = gldata.stich_images(y_img)
        save_images(image_id, img, mask, 'get')


save_with_get_data(source)
