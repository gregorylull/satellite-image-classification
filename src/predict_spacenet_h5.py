import keras.backend as K
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import glob

from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow, concatenate_images, imsave

import tensorflow as tf
from tqdm import tqdm
import re

from keras.models import load_model
from keras.layers import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

# glull files
from src.utilities import data as gldata
from src.utilities import unet as glunet

# https://stackoverflow.com/questions/55350010/ive-installed-cudnn-but-error-failed-to-get-convolution-algorithm-shows-up
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

USE_CACHED_MODEL = True

# play with this value to increase AUC
preds_threshold = 0.5

CHECK_IMAGES = 20

SAVE_FIG = True

# Set some parameters
# image_dimension = 650 # original shape
split_dimension = 128
image_dimension = 640  # resized dimension
split_len = int(image_dimension // split_dimension)
square_dim = split_len ** 2
image_channels = 3
mask_channels = 1
RGB_bits = 255  # RGB images
mask_bits = 255  # grayscale

# train_percentage = 1  # 1
# train_percentage = 0.75  # 1
train_percentage = 0.50  # 1
# train_percentage = 0.25  # 1
# train_percentage = 0.1  # 1
# train_percentage = 0.05  # 1
# train_percentage = 0.001  # 1

# model
batch_size = 64  # 32
dropout = 0.1  # 0.1
n_filters = 32  # 16
epochs = 50  # 100
patience = 15  # 20
float_type = 'float16'

EXT = '.png'

# path_train = 'data/salt/gltrain/
assets = 'assets/xview2/'

data = 'data/xview2/xview2_tier3/'

image_folder = 'snippet'

split_image_folder = 'snippet_split128_bit255_float16'

IMAGE_DIR = os.path.join(data, image_folder)

IMAGE_SPLIT_DIR = os.path.join(data, split_image_folder)

image_glob_paths = os.path.join(IMAGE_SPLIT_DIR, f'*{EXT}')

image_paths = sorted(glob.glob(image_glob_paths))

# Get and resize train images and masks

parameters = f'_epochs{epochs}_train{train_percentage}_batch{batch_size}_dropout{dropout}_n_filters{n_filters}_{float_type}'
CACHED_MODEL_FILENAME = os.path.join('models', f'spacenet{parameters}.h5')
CACHED_PREDICTION_FILENAME = os.path.join(
    'models', f'snippet_spacenet{parameters}_preds.pkl')


def savefig(fig, name, save=SAVE_FIG):
    if save:
        fig.savefig(
            name,
            bbox_inches='tight'
        )

# Split train and valid
# TODO after this split the variable id_without_ext will be off, and also the y_train...?


def get_images(image_paths):
    results = np.zeros((len(image_paths), split_dimension,
                        split_dimension, image_channels))

    for index, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
        img = imread(image_path)
        results[index, ...] = img

    return results


def create_model(
    split_dimension,
    image_channels,
    n_filters,
    dropout,
    loss='binary_crossentropy',
    metrics=['accuracy'],
    optimizer=Adam,
    batchnorm=True
):
    input_img = Input((split_dimension, split_dimension,
                       image_channels), name='img')
    model = glunet.get_unet(input_img, n_filters=n_filters,
                            dropout=dropout, batchnorm=batchnorm)

    model.compile(
        optimizer=optimizer(),
        loss=loss,
        metrics=metrics
    )

    model.summary()

    print('model summary')

    return model


def load_model(
    model,
    cached_model_filename=CACHED_MODEL_FILENAME,
    parameters=parameters
):

    print(f'\n\nUsing cached model: {cached_model_filename}')
    model.load_weights(cached_model_filename)

    return model


# Predict on train, val and test


def get_predictions(model, X_test, cached_prediction_filename=CACHED_PREDICTION_FILENAME):
    if os.path.exists(cached_prediction_filename):
        with open(cached_prediction_filename, 'rb') as readfile:
            preds = pickle.load(readfile)
            preds_test = preds['test']

    else:
        preds_test = model.predict(X_test, verbose=1)
        preds = {
            'test': preds_test
        }

        with open(cached_prediction_filename, 'wb') as writefile:
            pickle.dump(preds, writefile)

    return preds

# Threshold predictions


def get_predictions_thresholds(preds, preds_threshold=preds_threshold):
    preds_test = preds['test']

    preds_test_t = (preds_test > preds_threshold).astype(np.uint8)

    return preds_test_t


def plot_sample(images, preds, binary_preds, ix=None):
    post_index = ix

    post_start = post_index * square_dim
    post_end = post_start + square_dim

    # original images
    post_image = gldata.stich_images(
        images[post_start:post_end]).astype(np.int)

    # predicted images
    post_binary_preds_image = gldata.stich_images(
        binary_preds[post_start:post_end]).astype(np.float32)

    plt.close()

    fig, ax = plt.subplots(
        1, 2, figsize=(20, 10)
    )

    ax[0].imshow(post_image)
    ax[1].imshow(post_binary_preds_image.squeeze(), vmin=0, vmax=1)

    return fig


def output_train_val_test_images(
    images,
    preds,
    preds_threshold=preds_threshold,
    split_len=split_len,
    parameters=parameters
):

    dirpath = os.path.join(assets, parameters)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    preds_test = preds['test']

    preds_test_t = get_predictions_thresholds(
        preds, preds_threshold)

    # Check if training data looks all right

    # Check tests
    for i in range(min(CHECK_IMAGES, int(images.shape[0] / (split_len ** 2)))):
        fig_val = plot_sample(images, preds_test, preds_test_t, ix=i)
        name = f'{dirpath}/snippet_spacenet_test_predicted_fig_{i}_{parameters}.png'
        savefig(fig_val, name)


def main():

    model_params = []

    read_files_start = time.time()

    images = get_images(image_paths)

    read_files_end = time.time() - read_files_start

    # check_input_images(X_train, y_train, parameters)

    # load model
    model = create_model(
        split_dimension,
        image_channels,
        n_filters,
        dropout,
    )

    model = load_model(
        model,
        CACHED_MODEL_FILENAME,
        parameters
    )

    # get predictions
    predict_output_image_start = time.time()

    preds = get_predictions(
        model,
        images,
        CACHED_PREDICTION_FILENAME
    )

    output_train_val_test_images(
        images,
        preds,
        preds_threshold,
        split_len,
        parameters
    )

    predict_output_image_end = time.time() - predict_output_image_start

    total_per = (read_files_end +
                 predict_output_image_end) // 60
    prefix = f'minutes; Total_per ({len(image_paths) // square_dim}): {total_per}, Read: {read_files_end // 60 },  predict/output: {predict_output_image_end // 60}'

    print(f'\n\nparameters:\n{parameters}\n\n')

    model_params.append(
        (total_per, prefix, CACHED_PREDICTION_FILENAME)
    )

    return model_params


if __name__ == '__main__':
    model_params = main()
    print('\n\n\nParameters:\n', model_params)
