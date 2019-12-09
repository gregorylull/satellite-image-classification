
import os
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm_notebook, tnrange, tqdm

from skimage.io import imread, imshow, concatenate_images, imsave
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from src.utilities import data as gldata

import pickle
import re

data = 'data/xview2/xview2_tier3/'

# image_folder = 'snippet_test_images'

# testing for spacenet
image_folder = 'snippet'

IMAGE_DIR = os.path.join(data, image_folder)

image_glob_paths = os.path.join(IMAGE_DIR, '*.png')

split_dimension = 128
# image_dimension = 1024
image_dimension = 640  # testing for spacenet
image_channels = 3
# mask_channels = 1
RGB_bits = 255  # RGB images
# mask_bits = 255  # grayscale
EXT = '.png'

train_percentage = 1
# train_percentage = 0.1
snippet_image_paths = glob.glob(image_glob_paths)

train_index = int((len(snippet_image_paths) * train_percentage) // 1)

if not snippet_image_paths:
    print('\npath not correct\n')
    exit()

if train_percentage != 1:
    snippet_image_paths = snippet_image_paths[:train_index]


print(f'Splitting {len(snippet_image_paths)} images')


def create_data(
    image_paths,
    split_dimension=image_dimension,
    image_dimension=image_dimension,
    image_channels=image_channels,
):

    postfix = f'_split{split_dimension}_bit{RGB_bits}_float{16}'

    preprocessed_image_dir = os.path.join(data, f'{image_folder}{postfix}')

    use_preprocessed = os.path.exists(
        preprocessed_image_dir)

    if use_preprocessed:
        print('\nImage directories already exist, please deleted to create new images.\n')
        return

    os.mkdir(preprocessed_image_dir)

    print(f'creating preprocessed directory for {postfix} images')

    split_len = int(image_dimension // split_dimension)
    print(f'\nimages ({image_dimension}, {image_dimension}) will be split into ({split_dimension}, {split_dimension}) == {split_len ** 2} images\n')

    # sanity check
    assert (split_len * split_dimension) == image_dimension

    for image_path in tqdm(image_paths, total=len(image_paths)):

        # data/xview2/xview2_tier3/snippet_test_images/joplin-tornado_00000000_post_disaster.png
        filename = image_path.split('/')[-1]
        filename_wo_ext = re.sub(EXT, '', filename)

        # Load images
        x_img = imread(image_path)

        x_img = resize(x_img, (image_dimension, image_dimension, image_channels),
                       mode='constant', preserve_range=True)

        # commneted out for spacenet exp
        # x_img = x_img.squeeze() / RGB_bits
        # x_img = x_img.astype(np.float16)

        # Load masks
        x_split_images = gldata.split_image(
            x_img, image_dimension, split_dimension, split_len)

        for index, x_split_image in enumerate(x_split_images):
            part = 'part_' + str(index).zfill(2)
            image_save_path = os.path.join(
                preprocessed_image_dir, f'{filename_wo_ext}_{part}{EXT}')
            imsave(image_save_path, x_split_image)


if __name__ == '__main__':
    create_data(
        snippet_image_paths,
        split_dimension,
        image_dimension,
        image_channels,
    )
