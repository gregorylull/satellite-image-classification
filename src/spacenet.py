
import keras.backend as K
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split

import tensorflow as tf

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

train_percentage = 0.005  # 1
# train_percentage = 0.1  # 1

# train_percentage = 1  # 1

# play with this value to increase AUC
preds_threshold = 0.5

CHECK_IMAGES = 20

SAVE_FIG = True

# Set some parameters
# image_dimension = 650 # original shape
split_dimension = 128
image_dimension = 640  # resized dimension
image_channels = 3
mask_channels = 1

# model
batch_size = 16  # 32
dropout = 0.1  # 0.1
n_filters = 8  # 16
epochs = 50  # 100
patience = 10  # 20

# model FAST
batch_size = 8  # 32
dropout = 0.1  # 0.1
n_filters = 4  # 16
epochs = 30  # 100
patience = 10  # 20

# path_train = 'data/salt/gltrain/
path_train = 'data/salt/train/'
assets = 'assets/spacenet/'

# Get and resize train images and masks

parameters = f'_dim{split_dimension}_epochs{epochs}_train{train_percentage}_batch{batch_size}_dropout{dropout}_n_filters{n_filters}'
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
    ids, test_size=0.15)

X_ids_train, X_ids_valid = train_test_split(
    ids_remainder, test_size=0.20)

print(f'\n\nTrain percentage: {train_percentage * 100}%\n\n')

X_train, y_train = gldata.get_data(
    X_ids_train,
    split_dimension,
    image_dimension,
    image_channels,
    mask_channels,
)

X_valid, y_valid = gldata.get_data(
    X_ids_valid,
    split_dimension,
    image_dimension,
    image_channels,
    mask_channels,
)

X_test, y_test = gldata.get_data(
    ids_test,
    split_dimension,
    image_dimension,
    image_channels,
    mask_channels,
)


# Split train and valid
# TODO after this split the variable id_without_ext will be off, and also the y_train...?

# Check if training data looks all right
for ix in range(1):
    id_without_ext = X_ids_train[ix].split('.')[0]

    start = ix * 25
    end = start + 25

    x_image = gldata.stich_images(X_train[start:end])
    y_image = gldata.stich_images(y_train[start:end])
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
            axs[i, j].imshow(X_train[counter], interpolation='bilinear')
            axs[i, j].contour(y_train[counter].squeeze(),
                              colors='k', levels=[0.5])
            counter += 1

    counter = start + 0
    for i in range(5):
        for j in range(5, 10):
            axs[i, j].imshow(y_train[counter].squeeze(),
                             interpolation='bilinear', cmap='gray')
            counter += 1

    for ax_row in axs:
        for ax in ax_row:
            ax.label_outer()

    fig_name = os.path.join(
        assets, f'{id_without_ext}_unstiched_mask_comparison_{parameters}.png')
    print(f'\nSaving figure {fig_name}')
    savefig(fig, fig_name)


input_img = Input((split_dimension, split_dimension,
                   image_channels), name='img')
model = glunet.get_unet(input_img, n_filters=n_filters,
                        dropout=dropout, batchnorm=True)

model.compile(optimizer=Adam(), loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()

print('model summary')


if USE_CACHED_MODEL and os.path.exists(CACHED_MODEL_FILENAME):
    print(f'\n\nUsing cached model: {CACHED_MODEL_FILENAME}')
    model.load_weights(CACHED_MODEL_FILENAME)
    model.evaluate(X_valid, y_valid, verbose=1)

else:
    print(f'\n\nNo cached model found, will save as: {CACHED_MODEL_FILENAME}')

    callbacks = [
        EarlyStopping(patience=patience, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint(CACHED_MODEL_FILENAME, verbose=1,
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

# Predict on train, val and test

if os.path.exists(CACHED_PREDICTION_FILENAME):
    with open(CACHED_PREDICTION_FILENAME, 'rb') as readfile:
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

    with open(CACHED_PREDICTION_FILENAME, 'wb') as writefile:
        pickle.dump(preds, writefile)

# Threshold predictions
preds_train_t = (preds_train > preds_threshold).astype(np.uint8)
preds_val_t = (preds_val > preds_threshold).astype(np.uint8)
preds_test_t = (preds_test > preds_threshold).astype(np.uint8)


def plot_sample(X, y, preds, binary_preds, ix=None):
    start = ix * 25
    end = start + 25

    x_image = gldata.stich_images(X[start:end])

    y_image = gldata.stich_images(y[start:end])

    preds_image = gldata.stich_images(preds[start:end])

    binary_preds_image = gldata.stich_images(binary_preds[start:end])

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


# Check if training data looks all right
for i in range(min(CHECK_IMAGES, y_train.shape[0])):
    fig_train = plot_sample(
        X_train, y_train, preds_train, preds_train_t, ix=i)
    name = f'{assets}spacenet_train_predicted_fig_{i}_{parameters}.png'
    savefig(fig_train, name)

# Check if validation data looks all right
for i in range(min(CHECK_IMAGES, y_valid.shape[0])):
    fig_val = plot_sample(X_valid, y_valid, preds_val, preds_val_t, ix=i)
    name = f'{assets}spacenet_val_predicted_fig_{i}_{parameters}.png'
    savefig(fig_val, name)

# Check tests
for i in range(min(CHECK_IMAGES, y_test.shape[0])):
    fig_val = plot_sample(X_test, y_test, preds_test, preds_test_t, ix=i)
    name = f'{assets}spacenet_test_predicted_fig_{i}_{parameters}.png'
    savefig(fig_val, name)

print(f'\n\nparameters:\n{parameters}\n\n')
