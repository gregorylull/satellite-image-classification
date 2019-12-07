# (glull)
# This file is not directly related to my project, I am following a Medium post on how to use U-Net
# to identify salt segmentations to have a better understanding of image manipulation and neural nets.
# The data is from a Kaggle competition.
#
# NOTE, some of the code will be my own original code, and most of it will be from the medium post. Since this
# file is primarily for my own learning and data exploration I will not differentiate the code.
#
# medium post: https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47
# kaggle competition: https://www.kaggle.com/c/tgs-salt-identification-challenge/overview/description
# blog article (majority of the code from here): https://www.depends-on-the-definition.com/unet-keras-segmenting-images/

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
# float16 doesn't work
dtype_float = np.float32
# dtype_float = np.float16

CAST_TO_FLOAT16 = False

USE_CACHED_MODEL = True

TRAIN_PERCENTAGE = 1  # 1

CHECK_IMAGES = 5

CACHED_MODEL_FILENAME = f'model-tgs-salt_train{TRAIN_PERCENTAGE}.h5'

SAVE_FIG = True

# Set some parameters
im_width = 128
im_height = 128
border = 5
batch_size = 64  # 32
dropout = 0.1

# path_train = 'data/salt/gltrain/
path_train = 'data/salt/train/'
path_test = 'data/salt/'
assets = 'assets/salt/'

# Get and resize train images and masks


def savefig(fig, name, save=SAVE_FIG):
    if save:
        fig.savefig(name)


def get_data(path, train=True):
    ids = next(os.walk(path + "images"))[2]

    train_percentange = int((len(ids) * TRAIN_PERCENTAGE) // 1)
    ids = ids[:train_percentange]

    X = np.zeros((len(ids), im_height, im_width, 1), dtype=dtype_float)
    if train:
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=dtype_float)

    print('Getting and resizing images ... ')

    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load images
        img = load_img(path + '/images/' + id_, color_mode='grayscale')
        x_img = img_to_array(img)
        x_img = resize(x_img, (128, 128, 1),
                       mode='constant', preserve_range=True)

        # Load masks
        if train:
            mask = img_to_array(
                load_img(path + '/masks/' + id_, color_mode='grayscale'))
            mask = resize(mask, (128, 128, 1),
                          mode='constant', preserve_range=True)

        # Save images
        X[n, ..., 0] = x_img.squeeze() / 255
        if train:
            y[n] = mask / 255
    print('Done!')
    if train:
        return ids, X, y
    else:
        return ids, X


ids, X, y = get_data(path_train, train=True)

if CAST_TO_FLOAT16:
    X = tf.cast(X, tf.float16)
    y = tf.cast(y, tf.float16)

# Split train and valid
# TODO after this split the variable id_without_ext will be off, and also the y_train...?
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.15, random_state=2018)

# Check if training data looks all right
ix = random.randint(0, len(X_train) - 1)
id_without_ext = ids[ix].split('.')[0]
has_mask = y_train[ix].max() > 0

fig, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].imshow(X_train[ix, ..., 0], cmap='seismic', interpolation='bilinear')
if has_mask:
    ax[0].contour(y_train[ix].squeeze(), colors='k', levels=[0.5])
ax[0].set_title(f'Seismic {id_without_ext}')

ax[1].imshow(y_train[ix].squeeze(), interpolation='bilinear', cmap='gray')
ax[1].set_title(f'Salt {id_without_ext}')

fig_name = os.path.join(assets, f'{id_without_ext}_mask_comparison.png')
print(f'saving figure {fig_name}')
savefig(fig, fig_name)

# TODO this code isn't working as expected, the 2nd layer is overwriting the first layer
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
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


input_img = Input((im_height, im_width, 1), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

model.compile(optimizer=Adam(), loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()

print('model summary')

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint(CACHED_MODEL_FILENAME, verbose=1,
                    save_best_only=True, save_weights_only=True)
]

if USE_CACHED_MODEL:
    model.load_weights(CACHED_MODEL_FILENAME)
    model.evaluate(X_valid, y_valid, verbose=1)

else:

    epochs = 50  # 100
    results = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(X_valid, y_valid)
    )

    plt.figure(figsize=(12, 12))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(results.history["val_loss"]), np.min(
        results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()

    name = os.path.join(assets, 'tgs_learning_curve.png')
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
    ax[0].imshow(X[ix, ..., 0], cmap='seismic')
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Seismic')

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('Salt')

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Salt Predicted')

    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('Salt Predicted binary')

    return fig


# Check if training data looks all right
for i in range(CHECK_IMAGES):
    fig_train = plot_sample(
        X_train, y_train, preds_train, preds_train_t, ix=i)
    name = f'{assets}salt_train_predicted_fig_{i}_train{TRAIN_PERCENTAGE}.png'
    savefig(fig_train, name)

# Check if valid data looks all right
for i in range(CHECK_IMAGES):
    fig_val = plot_sample(X_valid, y_valid, preds_val, preds_val_t, ix=i)
    name = f'{assets}salt_val_predicted_fig_{i}_train{TRAIN_PERCENTAGE}.png'
    savefig(fig_val, name)
