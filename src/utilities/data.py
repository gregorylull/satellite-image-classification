import os
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm_notebook, tnrange, tqdm

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

IMAGE_DIR = os.path.join(data, AOI, 'RGB-PanSharpen')
GEOJSON_DIR = os.path.join(data, AOI, 'geojson', 'buildings')
OUTPUT_DIR = os.path.join(data, AOI, 'RGB-PanSharpen_masks')

vegas_csv = 'AOI_2_Vegas_Train_Building_Solutions.csv'

image_dimension = 650
image_channels = 3
mask_channels = 1
float_type = 'float16'

dtype_float = np.float32

RGB_bits = 2047  # RGB images
mask_bits = 255  # grayscale


def get_image_ids(csv_file=vegas_csv):

    image_id_filename = os.path.join(data, AOI, f'{csv_file}_ids.pkl')

    if os.path.exists(image_id_filename):
        with open(image_id_filename, 'rb') as readfile:
            loaded = pickle.load(readfile)
            print(f'\n\nUsing pickled ids #{len(loaded)}\n')
            return loaded

    else:
        print()
        summary_csv = os.path.join(
            data, AOI, 'summaryData', csv_file
        )
        df = pd.read_csv(summary_csv)
        imageids = df['ImageId'].unique()

        sorted_ids = sorted(imageids)

        existing_ids = []
        for id_ in sorted_ids:
            image_path = os.path.join(IMAGE_DIR, f'{image_prefix}{id_}.tif')
            mask_path = os.path.join(OUTPUT_DIR, f'{id_}_mask.tif')
            files_exist = (os.path.exists(image_path)
                           and os.path.exists(mask_path))
            if files_exist:
                existing_ids.append(id_)

        with open(image_id_filename, 'wb') as writefile:
            print(
                f'Caching ids that exist for image and mask: {len(existing_ids)} / {len(sorted_ids)} {len(existing_ids) / len(sorted_ids) * 100}%')
            pickle.dump(existing_ids, writefile)

        return sorted_ids


def create_data(
    ids,
    split_dimension=image_dimension,
    image_dimension=image_dimension,
    image_channels=image_channels,
    mask_channels=mask_channels,
    dtype_float=np.float32
):
    postfix = f'_resize{image_dimension}_split{split_dimension}_bit{RGB_bits}_float{dtype_float}'

    preprocessed_image_dir = f'{IMAGE_DIR}{postfix}'
    preprocessed_output_dir = f'{OUTPUT_DIR}{postfix}'

    if dtype_float == '32':
        float_type = np.float32
    elif dtype_float == '16':
        float_type = np.float16
    else:
        float_type = np.float32

    use_preprocessed = os.path.exists(
        preprocessed_image_dir) and os.path.exists(preprocessed_output_dir)

    if use_preprocessed:
        print('\nImage directories already exist, please deleted to create new images.\n')
        return

    os.mkdir(preprocessed_image_dir)
    os.mkdir(preprocessed_output_dir)

    print(f'creating preprocessed directory for {postfix} images')

    split_len = int(image_dimension // split_dimension)
    print(f'\nimages ({image_dimension}, {image_dimension}) will be split into ({split_dimension}, {split_dimension}) == {split_len ** 2} images\n')

    # sanity check
    assert (split_len * split_dimension) == image_dimension

    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        image_path = os.path.join(IMAGE_DIR, f'{image_prefix}{id_}.tif')
        mask_path = os.path.join(OUTPUT_DIR, f'{id_}_mask.tif')

        if not (os.path.exists(image_path) and os.path.exists(mask_path)):
            continue

        # Load images
        x_img = tifffile.imread(image_path)

        mask = tifffile.imread(mask_path)

        x_img = resize(x_img, (image_dimension, image_dimension, image_channels),
                       mode='constant', preserve_range=True)

        x_img = x_img.squeeze() / RGB_bits

        x_img = x_img.astype(float_type)

        # Load masks
        y_mask = resize(mask, (image_dimension, image_dimension, mask_channels),
                        mode='constant', preserve_range=True)

        y_mask = y_mask / mask_bits

        y_mask = y_mask.astype(float_type)

        x_split_images = split_image(
            x_img, image_dimension, split_dimension, split_len)
        y_split_images = split_image(
            y_mask, image_dimension, split_dimension, split_len)

        for index, x_split_image in enumerate(x_split_images):
            part = 'part_' + str(index).zfill(2)
            image_save_path = os.path.join(
                preprocessed_image_dir, f'{image_prefix}{id_}_{part}.tif')
            tifffile.imsave(image_save_path, x_split_image)

        for index, y_split_image in enumerate(y_split_images):
            part = 'part_' + str(index).zfill(2)
            mask_save_path = os.path.join(
                preprocessed_output_dir, f'{id_}_mask_{part}.tif')
            tifffile.imsave(mask_save_path, y_split_image)


def split_image(image, image_dimension, split_dimension, split_len):

    channels = image.shape[2]

    if split_len == 1:
        return [image]

    results = []

    linspace = np.linspace(0, image_dimension, split_len + 1).astype(np.int)
    for i, i_pixel in enumerate(linspace):
        if i == 0:
            continue
        start_i_pixel = linspace[i-1]

        for j, j_pixel in enumerate(linspace):
            if j == 0:
                continue
            start_j_pixel = linspace[j-1]
            result = image[start_i_pixel:i_pixel,
                           start_j_pixel:j_pixel, 0:channels]
            results.append(result)

    return results


def stich_images(images):
    split_dimension = images[0].shape[0]
    channels = images[0].shape[2]
    split_len = int(len(images) ** (1/2))
    original_dimension = split_dimension * split_len

    original = np.zeros((original_dimension, original_dimension, channels))

    linspace = np.linspace(0, original_dimension, split_len + 1).astype(np.int)
    counter = 0
    for i, i_pixel in enumerate(linspace):
        if i == 0:
            continue
        start_i_pixel = linspace[i-1]

        for j, j_pixel in enumerate(linspace):
            if j == 0:
                continue
            start_j_pixel = linspace[j-1]
            image = images[counter]
            original[start_i_pixel:i_pixel,
                     start_j_pixel:j_pixel, 0:channels] = image

            counter += 1

    return original


def stich_images_by_ids(ids):
    pass


def get_data(
    ids,
    split_dimension=image_dimension,
    image_dimension=image_dimension,
    image_channels=image_channels,
    mask_channels=mask_channels,
    float_type=float_type
):

    postfix = f'_resize{image_dimension}_split{split_dimension}_bit2047_{float_type}'
    preprocessed_image_dir = f'{IMAGE_DIR}{postfix}'
    preprocessed_output_dir = f'{OUTPUT_DIR}{postfix}'

    use_preprocessed = os.path.exists(
        preprocessed_image_dir) and os.path.exists(preprocessed_output_dir)

    if float_type == 'float16':
        dtype_float = np.float16
    elif float_type == 'float32':
        dtype_float = np.float32

    if use_preprocessed:
        split_len = int(image_dimension // split_dimension)
        print(f'Using preprocessed {postfix} images')
        image_dir = preprocessed_image_dir
        output_dir = preprocessed_output_dir
        total_ids = len(ids) * (split_len * split_len)
        X = np.zeros((total_ids, split_dimension, split_dimension,
                      image_channels), dtype=dtype_float)
        y = np.zeros((total_ids, split_dimension, split_dimension,
                      mask_channels), dtype=dtype_float)

        print(
            f'\nExamining images (use preprocess) {total_ids}')

    else:
        image_dir = IMAGE_DIR
        output_dir = OUTPUT_DIR

        # (650, 650) does not play well with pooling and upsampling concat
        image_dimension = 512
        X = np.zeros((len(ids), image_dimension, image_dimension,
                      image_channels), dtype=dtype_float)
        y = np.zeros((len(ids), image_dimension, image_dimension,
                      mask_channels), dtype=dtype_float)
        print(
            f'\nExamining images {len(ids)}')

    for n, id_ in tqdm(enumerate(ids), total=len(ids)):

        start = n * (split_len ** 2)

        if use_preprocessed:

            image_paths = os.path.join(
                image_dir, f'{image_prefix}{id_}_part_*.tif')
            mask_paths = os.path.join(output_dir, f'{id_}_mask_part_*.tif')

            if n == 0:
                print(f'\nExample path: {image_paths}')

            image_paths_glob = sorted(glob.glob(image_paths))
            mask_paths_glob = sorted(glob.glob(mask_paths))

            files_exist = image_paths_glob and mask_paths_glob
            if not files_exist:
                continue

            for image_count, image_path in enumerate(image_paths_glob):
                # Load images
                x_img = tifffile.imread(image_path)

                # TODO there may be an issue right here
                # fixed with dividing y_mask by a certain size.
                # x_norm = x_img.squeeze() / RGB_bits
                # X[start + image_count] = x_norm

                # data has already been normalized in create_data()
                X[start + image_count] = x_img

            for mask_count, mask_path in enumerate(mask_paths_glob):
                # Load masks
                y_mask = tifffile.imread(mask_path)
                # y_norm = y_mask / mask_bits
                # y[start + mask_count] = y_norm

                y[start + mask_count] = y_mask

        else:
            image_path = os.path.join(image_dir, f'{image_prefix}{id_}.tif')
            mask_path = os.path.join(output_dir, f'{id_}_mask.tif')
            files_exist = (os.path.exists(image_path)
                           and os.path.exists(mask_path))

            if n == 0:
                print(f'\nExample path: {image_path}')

            if not files_exist:
                continue

            # Load images
            x_img = tifffile.imread(image_path)

            # Load masks
            mask = tifffile.imread(mask_path)

            # REQUIRES RESIZING
            x_img = resize(x_img, (image_dimension, image_dimension, image_channels),
                           mode='constant', preserve_range=True)

            y_mask = resize(mask, (image_dimension, image_dimension, mask_channels),
                            mode='constant', preserve_range=True)

            x_norm = x_img.squeeze() / RGB_bits
            y_norm = y_mask / mask_bits

            X[n, ...] = x_norm
            y[n] = y_norm

    print('\nDone getting images!')
    return X, y


# X = np.zeros((8, 4, 4, 3))
# for i in range(4):
#     something = np.zeros((4, 4, 3))
#     something.fill(i + 1)
#     X[i] = something

# img = gldata.stich_images(X[0:4])
# img
