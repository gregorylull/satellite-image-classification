
import keras.backend as K
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# glull files
from src.utilities import data as gldata

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

USE_CACHED_MODEL = False

train_percentage = 0.02  # 1

CHECK_IMAGES = 10

SAVE_FIG = True

# Set some parameters
# image_dimension = 650 # original shape
image_dimension = 512 # divisible down the layers and easy to concatenate
image_channels = 3
mask_channels = 1

# model
border = 5
batch_size = 16  # 32
dropout = 0.1 # 0.1
n_filters = 8 # 16
epochs = 20 # 100
patience = 5 # 20

# path_train = 'data/salt/gltrain/
path_train = 'data/salt/train/'
assets = 'assets/spacenet/'

# Get and resize train images and masks

parameters = f'_train{train_percentage}_batch{batch_size}_dropout{dropout}_n_filters{n_filters}' 
CACHED_MODEL_FILENAME = f'models/spacenet{parameters}.h5'


def savefig(fig, name, save=SAVE_FIG):
    if save:
        fig.savefig(name)


ids, X, y = gldata.get_data(
    image_dimension, image_channels, mask_channels, train_percentage, train=True)

# Split train and valid
# TODO after this split the variable id_without_ext will be off, and also the y_train...?
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.15, random_state=2018)

# Check if training data looks all right
ix = random.randint(0, len(X_train) - 1)
id_without_ext = ids[ix].split('.')[0]
has_mask = y_train[ix].max() > 0

fig, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].imshow(X_train[ix, ..., 0], interpolation='bilinear')
# ax[0].imshow(X_train[ix, ..., 0], cmap='seismic', interpolation='bilinear')
if has_mask:
    ax[0].contour(y_train[ix].squeeze(), colors='k', levels=[0.5])
ax[0].set_title(f'Satellite {id_without_ext}')

ax[1].imshow(y_train[ix].squeeze(), interpolation='bilinear', cmap='gray')
ax[1].set_title(f'Buildings {id_without_ext}')

fig_name = os.path.join(assets, f'{id_without_ext}_mask_comparison.png')
print(f'\nSaving figure {fig_name}')
savefig(fig, fig_name)


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):

    """
    Function to add 2 convolutional layers with the parameters passed to it
    """
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def get_unet(input_img, n_filters=16, dropout=dropout, batchnorm=True):

    """
    create a unet model

    # (glull) keras didn't have a built-in U-Net model so the overall code for get_unet is from
    # an online source: https://www.depends-on-the-definition.com/unet-keras-segmenting-images/,
    # the only things i tweaked were the variables, dimensions, and filters.

    """
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1,
                      kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16,
                      kernel_size=3, batchnorm=batchnorm)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3),
                         strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3),
                         strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3),
                         strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3),
                         strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


input_img = Input((image_dimension, image_dimension,
                   image_channels), name='img')
model = get_unet(input_img, n_filters=n_filters, dropout=dropout, batchnorm=True)

model.compile(optimizer=Adam(), loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()

print('model summary')

callbacks = [
    EarlyStopping(patience=patience, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint(CACHED_MODEL_FILENAME, verbose=1,
                    save_best_only=True, save_weights_only=True)
]

if USE_CACHED_MODEL and os.path.exists(CACHED_MODEL_FILENAME):
    model.load_weights(CACHED_MODEL_FILENAME)
    model.evaluate(X_valid, y_valid, verbose=1)

else:

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
preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_valid, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)


def plot_sample(X, y, preds, binary_preds, ix=None):
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    plt.close()
    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0])
    # ax[0].imshow(X[ix, ..., 0], cmap='seismic')
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Satellite')

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('Buildings')

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Buildings Predicted')

    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('Buildings Predicted binary')

    return fig


# Check if training data looks all right
for i in range(CHECK_IMAGES):
    fig_train = plot_sample(
        X_train, y_train, preds_train, preds_train_t, ix=i)
    name = f'{assets}spacenet_train_predicted_fig_{i}{parameters}.png'
    savefig(fig_train, name)

# Check if validation data looks all right
for i in range(CHECK_IMAGES):
    fig_val = plot_sample(X_valid, y_valid, preds_val, preds_val_t, ix=i)
    name = f'{assets}spacenet_val_predicted_fig_{i}{parameters}.png'
    savefig(fig_val, name)
