
import keras.backend as K
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tqdm import tqdm

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

# set float16 https://medium.com/@noel_kennedy/how-to-use-half-precision-float16-when-training-on-rtx-cards-with-tensorflow-keras-d4033d59f9e4
# note kareas.json is in ~/.keras/
# dtype = 'float16'
# K.set_floatx(dtype)

# # default is 1e-7 which is too small for float16.  Without adjusting the epsilon, we will get NaN predictions because of divide by zero problems
# K.set_epsilon(1e-4)

# defualt is np.float32
# float16 doesn't work when saving matplotlib images
dtype_float = np.float32
# dtype_float = np.float16

USE_CACHED_MODEL = True


# train_percentage = 1  # 1

# play with this value to increase AUC
preds_threshold = 0.5

CHECK_IMAGES = 20

SAVE_FIG = True

# Set some parameters
# image_dimension = 650 # original shape
split_dimension = 128
image_dimension = 640  # resized dimension
split_len = int(image_dimension // split_dimension)
image_channels = 3
mask_channels = 1
RGB_bits = 2047  # RGB images
mask_bits = 255  # grayscale

train_percentage = 1  # 1
# train_percentage = 0.75  # 1
# train_percentage = 0.50  # 1
# train_percentage = 0.25  # 1
# train_percentage = 0.1  # 1
# train_percentage = 0.05  # 1
# train_percentage = 0.001  # 1

# model
batch_size = 64  # 32
dropout = 0.1  # 0.1
n_filters = 16  # 16
epochs = 50  # 100
patience = 15  # 20
float_type = 'float16'

# model medium
# batch_size = 32  # 32
# dropout = 0.1  # 0.1
# n_filters = 8  # 16
# epochs = 50  # 100
# patience = 20  # 20
# float_type = 'float16'

# model FAST
# batch_size = 8  # 32
# dropout = 0.1  # 0.1
# n_filters = 4  # 16
# epochs = 30  # 100
# patience = 10  # 20
# float_type = 'float16'

# path_train = 'data/salt/gltrain/
path_train = 'data/salt/train/'
assets = 'assets/spacenet/'

# Get and resize train images and masks

parameters = f'_epochs{epochs}_train{train_percentage}_batch{batch_size}_dropout{dropout}_n_filters{n_filters}_{float_type}'
CACHED_MODEL_FILENAME = f'models/spacenet{parameters}.h5'
CACHED_PREDICTION_FILENAME = f'models/spacenet{parameters}_preds.pkl'


def savefig(fig, name, save=SAVE_FIG):
    if save:
        fig.savefig(
            name,
            bbox_inches='tight'
        )


ids = gldata.get_image_ids()

if train_percentage != 1:
    train_index = int((len(ids) * train_percentage) // 1)
    ids = ids[:train_index]

ids_remainder, ids_test = train_test_split(
    ids, test_size=0.20)

X_ids_train, X_ids_valid = train_test_split(
    ids_remainder, test_size=0.20)

print(f'\n\nTrain percentage: {len(ids)} {train_percentage * 100}%\n\n')


def get_tvt(
    X_ids_train,
    X_ids_valid,
    ids_test,
    float_type,
    split_dimension=split_dimension,
    image_dimension=image_dimension,
    image_channels=image_channels,
    mask_channels=mask_channels
):
    X_train, y_train = gldata.get_data(
        X_ids_train,
        split_dimension,
        image_dimension,
        image_channels,
        mask_channels,
        float_type
    )

    X_valid, y_valid = gldata.get_data(
        X_ids_valid,
        split_dimension,
        image_dimension,
        image_channels,
        mask_channels,
        float_type
    )

    X_test, y_test = gldata.get_data(
        ids_test,
        split_dimension,
        image_dimension,
        image_channels,
        mask_channels,
        float_type
    )

    return X_train, y_train, X_valid, y_valid, X_test, y_test

# Split train and valid
# TODO after this split the variable id_without_ext will be off, and also the y_train...?


# Check if training data looks all right
def check_input_images(X_train, y_train, parameters):
    for ix in range(5):
        id_without_ext = X_ids_train[ix].split('.')[0]

        start = ix * 25
        end = start + 25

        x_image = gldata.stich_images(X_train[start:end]).astype(np.float32)
        y_image = gldata.stich_images(y_train[start:end]).astype(np.float32)
        has_mask = y_image.max() > 0

        fig, ax = plt.subplots(1, 2, figsize=(20, 10))

        ax[0].imshow(x_image, interpolation='bilinear')
        # ax[0].imshow(X_train[ix, ..., 0], cmap='seismic', interpolation='bilinear')
        if has_mask:
            ax[0].contour(y_image.squeeze(), colors='k', levels=[0.5])
        ax[0].set_title(f'Satellite {id_without_ext}')

        ax[1].imshow(y_image.squeeze(), interpolation='bilinear', cmap='gray')
        ax[1].set_title(f'Buildings {id_without_ext}')

        fig_name = os.path.join(
            assets, f'{id_without_ext}_mask_comparison_{parameters}.png')
        print(f'\nSaving figure {fig_name}\n')
        savefig(fig, fig_name)

        # check 25 images
        fig, axs = plt.subplots(
            5, 10,
            sharex=True, sharey=True,
            gridspec_kw={'hspace': 0, 'wspace': 0},
            figsize=(20, 10)
        )

        fig.suptitle('Satellite with Building Mask')
        counter = start + 0
        id_without_ext = ids[ix].split('.')[0]
        for i in range(5):
            for j in range(5):
                axs[i, j].imshow(X_train[counter].astype(
                    np.float32), interpolation='bilinear')
                axs[i, j].contour(y_train[counter].astype(np.float32).squeeze(),
                                  colors='k', levels=[0.5])
                counter += 1

        counter = start + 0
        for i in range(5):
            for j in range(5, 10):
                axs[i, j].imshow(y_train[counter].astype(np.float32).squeeze(),
                                 interpolation='bilinear', cmap='gray')
                counter += 1

        for ax_row in axs:
            for ax in ax_row:
                ax.label_outer()

        fig_name = os.path.join(
            assets, f'{id_without_ext}_stiched_mask_comparison_{parameters}.png')
        print(f'\nSaving figure {fig_name}')
        savefig(fig, fig_name)


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


def train_cache_model(
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    patience=patience,
    batch_size=batch_size,
    epochs=epochs,
    use_cached_model=USE_CACHED_MODEL,
    cached_model_filename=CACHED_MODEL_FILENAME,
    parameters=parameters
):
    if use_cached_model and os.path.exists(cached_model_filename):
        print(f'\n\nUsing cached model: {cached_model_filename}')
        model.load_weights(cached_model_filename)
        model.evaluate(X_valid, y_valid, verbose=1)

    else:
        print(
            f'\n\nNo cached model found, will save as: {cached_model_filename}')

        callbacks = [
            EarlyStopping(patience=patience, verbose=1),
            ReduceLROnPlateau(factor=0.1, patience=3,
                              min_lr=0.00001, verbose=1),
            ModelCheckpoint(cached_model_filename, verbose=1,
                            save_best_only=True, save_weights_only=True)
        ]

        epochs = epochs
        results = model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(X_valid, y_valid)
        )

        plt.figure(figsize=(12, 12))
        plt.title(f"Learning curve {' '.join(parameters.split('_'))}")
        plt.plot(results.history["loss"], label="loss")
        plt.plot(results.history["val_loss"], label="val_loss")
        plt.plot(np.argmin(results.history["val_loss"]), np.min(
            results.history["val_loss"]), marker="x", color="r", label="best model")
        plt.xlabel("Epochs")
        plt.ylabel("log_loss")
        plt.legend()

        name = os.path.join(assets, f'tgs_learning_curve{parameters}.png')
        savefig(plt, name)

    return model


# Predict on train, val and test


def get_predictions(model, X_train, X_valid, X_test, cached_prediction_filename=CACHED_PREDICTION_FILENAME):
    if os.path.exists(cached_prediction_filename):
        with open(cached_prediction_filename, 'rb') as readfile:
            preds = pickle.load(readfile)
            preds_train = preds['train']
            preds_val = preds['val']
            preds_test = preds['test']

    else:
        preds_train = model.predict(X_train, verbose=1)
        preds_val = model.predict(X_valid, verbose=1)
        preds_test = model.predict(X_test, verbose=1)
        preds = {
            'train': preds_train,
            'val': preds_val,
            'test': preds_test
        }

        with open(cached_prediction_filename, 'wb') as writefile:
            pickle.dump(preds, writefile)

    return preds

# Threshold predictions


def get_predictions_thresholds(preds, preds_threshold=preds_threshold):
    preds_train = preds['train']
    preds_val = preds['val']
    preds_test = preds['test']

    preds_train_t = (preds_train > preds_threshold).astype(np.uint8)
    preds_val_t = (preds_val > preds_threshold).astype(np.uint8)
    preds_test_t = (preds_test > preds_threshold).astype(np.uint8)

    return preds_train_t, preds_val_t, preds_test_t


def plot_sample(X, y, preds, binary_preds, ix=None):
    start = ix * 25
    end = start + 25

    x_image = gldata.stich_images(X[start:end]).astype(np.float32)

    y_image = gldata.stich_images(y[start:end]).astype(np.float32)

    preds_image = gldata.stich_images(preds[start:end]).astype(np.float32)

    binary_preds_image = gldata.stich_images(
        binary_preds[start:end]).astype(np.float32)

    has_mask = y_image.max() > 0

    plt.close()

    fig, ax = plt.subplots(
        1, 4, figsize=(20, 10)
    )

    ax[0].imshow(x_image)
    if has_mask:
        ax[0].contour(y_image.squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Satellite')

    ax[1].imshow(y_image.squeeze())
    ax[1].set_title('Buildings')

    ax[2].imshow(preds_image.squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y_image.squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Buildings Predicted')

    ax[3].imshow(binary_preds_image.squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y_image.squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('Buildings Predicted binary')

    return fig


def output_train_val_test_images(
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_test,
    y_test,
    preds,
    preds_threshold=preds_threshold,
    split_len=split_len,
    parameters=parameters
):

    dirpath = os.path.join(assets, parameters)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    preds_train = preds['train']
    preds_val = preds['val']
    preds_test = preds['test']

    preds_train_t, preds_val_t, preds_test_t = get_predictions_thresholds(
        preds, preds_threshold)

    # Check if training data looks all right
    for i in range(min(CHECK_IMAGES, int(y_train.shape[0] / (split_len ** 2)))):
        fig_train = plot_sample(
            X_train, y_train, preds_train, preds_train_t, ix=i)
        name = f'{dirpath}/spacenet_train_predicted_fig_{i}_{parameters}.png'
        savefig(fig_train, name)

    # Check if validation data looks all right
    for i in range(min(CHECK_IMAGES, int(y_valid.shape[0] / (split_len ** 2)))):
        fig_val = plot_sample(X_valid, y_valid, preds_val, preds_val_t, ix=i)
        name = f'{dirpath}/spacenet_val_predicted_fig_{i}_{parameters}.png'
        savefig(fig_val, name)

    # Check tests
    for i in range(min(CHECK_IMAGES, int(y_test.shape[0] / (split_len ** 2)))):
        fig_val = plot_sample(X_test, y_test, preds_test, preds_test_t, ix=i)
        name = f'{dirpath}/spacenet_test_predicted_fig_{i}_{parameters}.png'
        savefig(fig_val, name)


def main_fast():

    batch_size = 16  # 32
    dropout = 0.1  # 0.1
    n_filters = 8  # 16
    epochs = 30  # 100
    patience = 10  # 20
    float_type = 'float16'

    model_params = []

    read_files_start = time.time()

    X_train, y_train, X_valid, y_valid, X_test, y_test = get_tvt(
        X_ids_train, X_ids_valid, ids_test, float_type)

    read_files_end = time.time() - read_files_start

    train_model_start = time.time()

    parameters = f'_epochs{epochs}_train{train_percentage}_batch{batch_size}_dropout{dropout}_n_filters{n_filters}_{float_type}'
    CACHED_MODEL_FILENAME = f'models/spacenet{parameters}.h5'
    CACHED_PREDICTION_FILENAME = f'models/spacenet{parameters}_preds.pkl'

    # check_input_images(X_train, y_train, parameters)

    model = create_model(
        split_dimension,
        image_channels,
        n_filters,
        dropout,
    )

    model = train_cache_model(
        model,
        X_train,
        y_train,
        X_valid,
        y_valid,
        patience,
        batch_size,
        epochs,
        USE_CACHED_MODEL,
        CACHED_MODEL_FILENAME,
        parameters
    )

    train_model_end = time.time() - train_model_start

    predict_output_image_start = time.time()

    preds = get_predictions(
        model,
        X_train,
        X_valid,
        X_test,
        CACHED_PREDICTION_FILENAME
    )

    output_train_val_test_images(
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        preds,
        preds_threshold,
        split_len,
        parameters
    )

    predict_output_image_end = time.time() - predict_output_image_start

    total_per = (read_files_end + train_model_end +
                 predict_output_image_end) // 60
    prefix = f'minutes; Total_per: {total_per}, Read: {read_files_end // 60 }, train: {train_model_end // 60}, predict/output: {predict_output_image_end // 60}'

    print(f'\n\nparameters:\n{parameters}\n\n')

    model_params.append(
        (total_per, prefix, CACHED_PREDICTION_FILENAME))

    return model_params


def main_grid():

    model_params = []

    for float_type in ['float16']:

        read_files_start = time.time()

        X_train, y_train, X_valid, y_valid, X_test, y_test = get_tvt(
            X_ids_train, X_ids_valid, ids_test, float_type)

        read_files_end = time.time() - read_files_start

        for dropout in tqdm([0.1, 0.2, 0.3, 0.4], total=3):

            for n_filters in [16]:

                for batch_size in [64]:

                    train_model_start = time.time()

                    parameters = f'_epochs{epochs}_train{train_percentage}_batch{batch_size}_dropout{dropout}_n_filters{n_filters}_{float_type}'
                    CACHED_MODEL_FILENAME = f'models/spacenet{parameters}.h5'
                    CACHED_PREDICTION_FILENAME = f'models/spacenet{parameters}_preds.pkl'

                    # check_input_images(X_train, y_train, parameters)

                    model = create_model(
                        split_dimension,
                        image_channels,
                        n_filters,
                        dropout,
                    )

                    model = train_cache_model(
                        model,
                        X_train,
                        y_train,
                        X_valid,
                        y_valid,
                        patience,
                        batch_size,
                        epochs,
                        USE_CACHED_MODEL,
                        CACHED_MODEL_FILENAME,
                        parameters
                    )

                    train_model_end = time.time() - train_model_start

                    predict_output_image_start = time.time()

                    preds = get_predictions(
                        model,
                        X_train,
                        X_valid,
                        X_test,
                        CACHED_PREDICTION_FILENAME
                    )

                    output_train_val_test_images(
                        X_train,
                        y_train,
                        X_valid,
                        y_valid,
                        X_test,
                        y_test,
                        preds,
                        preds_threshold,
                        split_len,
                        parameters
                    )

                    predict_output_image_end = time.time() - predict_output_image_start

                    total_per = (read_files_end + train_model_end +
                                 predict_output_image_end) // 60
                    prefix = f'minutes; Total_per: {total_per}, Read: {read_files_end // 60 }, train: {train_model_end // 60}, predict/output: {predict_output_image_end // 60}'

                    print(f'\n\nparameters:\n{parameters}\n\n')

                    combo = (total_per, prefix, CACHED_PREDICTION_FILENAME)
                    model_params.append(combo)

                    print(model_params)

    return model_params


if __name__ == '__main__':
    model_params = main_grid()
    # model_params = main_fast()
    print('\n\n\nParameters:\n', model_params)
