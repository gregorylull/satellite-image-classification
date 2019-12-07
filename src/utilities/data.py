import os
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook, tnrange

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from skimage.external import tifffile

import pickle

data = 'data/spacenet/'
AOI = 'AOI_2_Vegas_Train/'
image_prefix = 'RGB-PanSharpen_'
geojson_prefix = 'buildings_'
mask_postfix = '_mask'

image_dir = os.path.join(data, AOI, 'RGB-PanSharpen')
geojson_dir = os.path.join(data, AOI, 'geojson', 'buildings')
output_dir = os.path.join(data, AOI, 'RGB-PanSharpen_masks')

vegas_csv = 'AOI_2_Vegas_Train_Building_Solutions.csv'

image_dimension = 650
image_channels = 3
mask_channels = 1

dtype_float = np.float32

TRAIN_PERCENTAGE = 0.1
RGB_bits = 2047  # RGB images
mask_bits = 255  # grayscale


def get_image_ids(csv_file=vegas_csv):

    image_id_filename = os.path.join(data, AOI, f'{csv_file}_ids.pkl')

    if os.path.exists(image_id_filename):
        with open(image_id_filename, 'rb') as readfile:
            return pickle.load(readfile)

    else:
        summary_csv = os.path.join(
            data, AOI, 'summaryData', csv_file
        )
        df = pd.read_csv(summary_csv)
        imageids = df['ImageId'].unique()

        sorted_ids = sorted(imageids)

        with open(image_id_filename, 'wb') as writefile:
            pickle.dump(sorted_ids, writefile)

        return sorted_ids


def get_data(
    image_dimension=image_dimension,
    image_channels=image_channels,
    mask_channels=mask_channels,
    train_percentage=1.0,
    train=True
):
    total_ids = get_image_ids(vegas_csv)
    train_index = int((len(total_ids) * train_percentage) // 1)
    ids = total_ids[:train_index]

    print(
        f'\nExamining images {len(ids)} / {len(total_ids)} ({train_percentage}%)')

    X = np.zeros((len(ids), image_dimension, image_dimension,
                  image_channels), dtype=dtype_float)
    if train:
        y = np.zeros((len(ids), image_dimension, image_dimension,
                      mask_channels), dtype=dtype_float)

    print('Getting and resizing images ... ')

    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        image_path = os.path.join(image_dir, f'{image_prefix}{id_}.tif')
        mask_path = os.path.join(output_dir, f'{id_}_mask.tif')

        if not (os.path.exists(image_path) and os.path.exists(mask_path)):
            continue

        # Load images
        x_img = tifffile.imread(image_path)

        # x_img = img_to_array(img)
        x_img = resize(x_img, (image_dimension, image_dimension, image_channels),
                       mode='constant', preserve_range=True)

        # Load masks
        if train:
            mask = tifffile.imread(mask_path)
            # mask = img_to_array(
            #     load_img(path + '/masks/' + id_, color_mode='grayscale'))
            y_mask = resize(mask, (image_dimension, image_dimension, mask_channels),
                            mode='constant', preserve_range=True)

        # Save images
        # TODO there may be an issue right here
        X[n, ...] = x_img.squeeze() / RGB_bits
        if train:
            y[n] = y_mask / mask_bits
    print('Done!')
    if train:
        return ids, X, y
    else:
        return ids, X
