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
import pickle

from PIL import Image

# data/xview2/xview2_tier3/xBD/joplin-tornado/

# images/

# joplin-tornado_00000000_post_disaster.png
# joplin-tornado_00000000_post_disaster.png

data = 'data/xview2/xview2_tier3/xBD'
AOI = 'joplin-tornado'

IMAGE_DIR = os.path.join(data, AOI, 'images')
MASK_DIR = os.path.join(data, AOI, 'masks')

vegas_csv = 'AOI_2_Vegas_Train_Building_Solutions.csv'

image_dimension = 1024
image_channels = 3
mask_channels = 1
float_type = 'float16'

RGB_bits = 255  # RGB images
mask_bits = 255  # grayscale

ext = '.png'

AOIs = [
    'joplin-tornado',
    'lower-puna-volcano',
    'moore-tornado',
    'nepal-flooding',
    'pinery-bushfire',
    'portugal-widlfire',
    'sunda-tsunami',
    'tuscaloosa-tornado',
    'woolsey-fire'
]


def get_parameters(
    epochs,
    train_percentage,
    batch_size,
    dropout,
    n_filters,
    metric_params,
    loss_params,
    float_type,
):
    """
        return f'_epochs{epochs}_train{train_percentage}_batch{batch_size}_dropout{dropout}_n_filters{n_filters}_{float_type}'
    """

    parameters = [
        '',
        f'epochs{epochs}',
        f'train{train_percentage}',
        f'batch{batch_size}',
        f'dropout{dropout}',
        f'n_filters{n_filters}',
        f'metric-{metric_params}',
        f'loss-{loss_params}',
        f'{float_type}'
    ]
    return '_'.join(parameters)


def get_percentage(arr, percentage=1.0):
    if percentage != 1:
        index = int((len(arr) * percentage) // 1)
        arr = arr[:index]

    return arr


def get_image_ids(segmentation='all', train_percentage=1.0, test_size=0.2, AOI='joplin-tornado'):
    """
    inputs:
        segmentation = 'all', train', 'valid', 'test', 'remainder'(train + valid)

    returns:
        all the image ids for that segment
    """

    def get_segementation_path(AOI, segmentation, test_size):
        image_id_filename = os.path.join(
            data, AOI, f'{AOI}_{segmentation}_{test_size}_ids.pkl')
        return image_id_filename

    image_id_filename = get_segementation_path(
        AOI, segmentation, test_size)

    if os.path.exists(image_id_filename):
        with open(image_id_filename, 'rb') as readfile:
            loaded = pickle.load(readfile)
            percentage_loaded = get_percentage(loaded, train_percentage)
            print(
                f'\nUsing pickled ids ({segmentation} {train_percentage * 100})% {len(percentage_loaded)} / {len(loaded)}\n')
            return percentage_loaded

    else:
        print('\n\nSplitting ids into train valid test\n')
        image_path_globs = os.path.join(
            data, AOI, 'images', f'*{ext}'
        )

        image_paths = glob.glob(image_path_globs)

        imageids = [image_path.split('/')[-1] for image_path in image_paths]

        sorted_ids = sorted(imageids)

        ids_remainder, ids_test = train_test_split(
            sorted_ids, test_size=test_size)

        X_ids_train, X_ids_valid = train_test_split(
            ids_remainder, test_size=test_size)

        id_groups = {
            'all': sorted_ids,
            'train': X_ids_train,
            'valid': X_ids_valid,
            'remainder': ids_remainder,
            'test': ids_test
        }

        for id_group_key, id_group in id_groups.items():
            image_id_filename = get_segementation_path(
                AOI, id_group_key, test_size)

            with open(image_id_filename, 'wb') as writefile:
                print(
                    f'Caching ids that exist for {id_group_key} image and mask: {len(id_group)} / {len(sorted_ids)} {len(id_group) / len(sorted_ids) * 100}%')
                pickle.dump(id_group, writefile)

        return get_image_ids(segmentation, train_percentage, test_size, AOI)


def create_data(
    ids,
    split_dimension=image_dimension,
    image_dimension=image_dimension,
    image_channels=image_channels,
    mask_channels=mask_channels,
    dtype_float='16'
):
    postfix = f'_resize{image_dimension}_split{split_dimension}_bit{RGB_bits}_float{dtype_float}'

    preprocessed_image_dir = f'{IMAGE_DIR}{postfix}'
    preprocessed_MASK_DIR = f'{MASK_DIR}{postfix}'

    float_type = np.float16

    use_preprocessed = os.path.exists(
        preprocessed_image_dir) and os.path.exists(preprocessed_MASK_DIR)

    if use_preprocessed:
        print('\nImage directories already exist, please deleted to create new images.\n')
        return

    os.mkdir(preprocessed_image_dir)
    os.mkdir(preprocessed_MASK_DIR)

    print(f'creating preprocessed directory for {postfix} images')

    split_len = int(image_dimension // split_dimension)
    print(f'\nimages ({image_dimension}, {image_dimension}) will be split into ({split_dimension}, {split_dimension}) == {split_len ** 2} images\n')

    # sanity check
    assert (split_len * split_dimension) == image_dimension

    print(f'Splitting ids {len(ids)}')
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        image_path = os.path.join(IMAGE_DIR, f'{id_}{ext}')
        mask_path = os.path.join(MASK_DIR, f'{id_}{ext}')

        if not (os.path.exists(image_path) and os.path.exists(mask_path)):
            continue

        # Load images
        x_img_obj = Image.open(img_path)
        x_img = np.array(img_obj)

        x_img = x_img.squeeze() / RGB_bits
        x_img = x_img.astype(float_type)

        # Load masks
        mask_img_obj = Image.open(mask_path)
        mask = np.array(mask_img_obj)

        y_mask = y_mask / mask_bits
        y_mask = y_mask.astype(float_type)

        x_split_images = split_image(
            x_img, image_dimension, split_dimension, split_len)

        y_split_images = split_image(
            y_mask, image_dimension, split_dimension, split_len)

        for index, x_split_image in enumerate(x_split_images):
            part = 'part_' + str(index).zfill(2)
            image_save_path = os.path.join(
                preprocessed_image_dir, f'{id_}_{part}{ext}')
            Image.save(image_save_path, x_split_image)

        for index, y_split_image in enumerate(y_split_images):
            part = 'part_' + str(index).zfill(2)
            mask_save_path = os.path.join(
                preprocessed_MASK_DIR, f'{id_}_{part}{ext}')
            Image.save(mask_save_path, y_split_image)


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
    float_type='float16'
):

    postfix = f'_resize{image_dimension}_split{split_dimension}_bit{RGB_bits}_{float_type}'
    preprocessed_image_dir = f'{IMAGE_DIR}{postfix}'
    preprocessed_MASK_DIR = f'{MASK_DIR}{postfix}'

    use_preprocessed = os.path.exists(
        preprocessed_image_dir) and os.path.exists(preprocessed_MASK_DIR)

    dtype_float = np.float16

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

        image_paths = os.path.join(
            image_dir, f'{id_}_part_*{ext}')
        mask_paths = os.path.join(output_dir, f'{id_}_part_*{ext}')

        if n == 0:
            print(f'\nExample path: {image_paths}')

        image_paths_glob = sorted(glob.glob(image_paths))
        mask_paths_glob = sorted(glob.glob(mask_paths))

        files_exist = image_paths_glob and mask_paths_glob
        if not files_exist:
            continue

        for image_count, image_path in enumerate(image_paths_glob):
            # Load images
            x_img_obj = Image.open(image_path)
            x_img = np.array(x_img_obj)
            X[start + image_count] = x_img

        for mask_count, mask_path in enumerate(mask_paths_glob):
            y_mask_obj = Image.open(mask_path)
            y_mask = np.array(y_mask_obj)

            y[start + mask_count] = y_mask

    print('\nDone getting images!')
    return X, y


# X = np.zeros((8, 4, 4, 3))
# for i in range(4):
#     something = np.zeros((4, 4, 3))
#     something.fill(i + 1)
#     X[i] = something

# img = gldata.stich_images(X[0:4])
# img
