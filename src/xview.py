
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
from src.utilities import data_xview as gldata
from src.utilities import unet as glunet

# https://stackoverflow.com/questions/55350010/ive-installed-cudnn-but-error-failed-to-get-convolution-algorithm-shows-up
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

USE_CACHED_MODEL = True

# train_percentage = 1  # 1

# play with this value to increase AUC
preds_threshold = 0.5

CHECK_IMAGES = 20

SAVE_FIG = True

# Set some parameters
# image_dimension = 650 # original shape
split_dimension = 128
image_dimension = 1024  # resized dimension
split_len = int(image_dimension // split_dimension)
square_dim = split_len * split_len
image_channels = 3
mask_channels = 1
RGB_bits = 255  # RGB images
mask_bits = 255  # grayscale

train_percentage = 1  # 1
# train_percentage = 0.75  # 1
# train_percentage = 0.5  # 1
# train_percentage = 0.25  # 1
# train_percentage = 0.5  # 1
# train_percentage = 0.05  # 1
# train_percentage = 0.01  # 1
# train_percentage = 0.005  # 1

# model
batch_size = 64  # 32
dropout = 0.1  # 0.1
n_filters = 16  # 16
epochs = 80  # 100
patience = 15  # 20
float_type = 'float16'
# metrics = ['accuracy']  # ['accuracy']
# metric_params = 'accuracy'  # 'accuracy'
# loss_metrics = 'binary_crossentropy'  # 'binary_crossentropy'
# loss_params = 'binary_crossentropy'  # 'binary_crossentropy'

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

assets = 'assets/xview2/'

# Get and resize train images and masks

# parameters = gldata.get_parameters(
#     epochs,
#     train_percentage,
#     batch_size,
#     dropout,
#     n_filters,
#     metric_params,
#     loss_params,
#     float_type,
# )
# CACHED_MODEL_FILENAME = f'models/xview2{parameters}.h5'
# CACHED_PREDICTION_FILENAME = f'models/xview2{parameters}_preds.pkl'
# CACHED_EVALUATION_FILENAME = f'models/xview2{parameters}_evals.pkl'


def savefig(fig, name, save=SAVE_FIG):
    if save:
        fig.savefig(
            name,
            bbox_inches='tight'
        )


ids = gldata.get_image_ids('all', train_percentage)
X_ids_train = gldata.get_image_ids('train', train_percentage)
X_ids_valid = gldata.get_image_ids('valid', train_percentage)
ids_test = gldata.get_image_ids('test', train_percentage)
ids_remainder = gldata.get_image_ids('remainder', train_percentage)

USE_X_TRAIN = True
USE_X_VALID = True
USE_X_TEST = True

#
USE_X_TRAIN = False
USE_X_VALID = False
USE_X_TEST = True

print(
    f'\n\nTrain percentage ({train_percentage * 100}%): {len(ids) // 1} / {len(ids) * (1/train_percentage) // 1} \n\n')


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

    X_train, y_train = [], []
    X_valid, y_valid = [], []
    X_test, y_test = [], []

    if USE_X_TRAIN:
        X_train, y_train = gldata.get_data(
            X_ids_train,
            split_dimension,
            image_dimension,
            image_channels,
            mask_channels,
            float_type
        )

    if USE_X_VALID:
        X_valid, y_valid = gldata.get_data(
            X_ids_valid,
            split_dimension,
            image_dimension,
            image_channels,
            mask_channels,
            float_type
        )

    if USE_X_TEST:
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

        start = ix * square_dim
        end = start + square_dim

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
    metrics,
    loss,
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
    patience,
    batch_size,
    epochs,
    use_cached_model,
    cached_model_filename,
    parameters
):
    if use_cached_model and os.path.exists(cached_model_filename):
        print(f'\n\nUsing cached model: {cached_model_filename}')
        model.load_weights(cached_model_filename)

    else:
        print(
            f'\n\nNo cached model found, will save as: {cached_model_filename}')

        # (glull) tutorial and documentation
        # https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
        # https://keras.io/callbacks/#modelcheckpoint
        # custom metric https://stackoverflow.com/questions/43782409/how-to-use-modelcheckpoint-with-custom-metrics-in-keras
        callbacks = [
            EarlyStopping(
                patience=patience,
                verbose=1
            ),
            ReduceLROnPlateau(
                factor=0.1, patience=3,
                min_lr=0.00001, verbose=1
            ),
            ModelCheckpoint(
                cached_model_filename, verbose=1,
                save_best_only=True, save_weights_only=True,
                monitor='val_loss',
                mode='min'

            )
        ]

        xy_train_valid = np.array([
            X_train.shape[0], y_train.shape[0], X_valid.shape[0], y_valid.shape[0]])
        if not xy_train_valid.all():
            print(
                f'\n\nError modeling: X/y_train/valid are empty {xy_train_valid}')
            exit()

        epochs = epochs
        results = model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(X_valid, y_valid)
        )

        plt.figure(figsize=(15, 15))
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


def get_predictions(model, X_train, X_valid, X_test, cached_prediction_filename):

    if os.path.exists(cached_prediction_filename):
        try:
            with open(cached_prediction_filename, 'rb') as readfile:
                preds = pickle.load(readfile)
                print('\nUsing cached prediction files\n',
                      cached_prediction_filename)
                return preds
        except:
            print('Error opening cached prediction file\n',
                  cached_prediction_filename)

    preds_train = []
    preds_val = []
    preds_test = []

    if USE_X_TRAIN:
        preds_train = model.predict(X_train, verbose=1)

    if USE_X_VALID:
        preds_val = model.predict(X_valid, verbose=1)

    if USE_X_TEST:
        preds_test = model.predict(X_test, verbose=1)

    preds = {
        'train': preds_train,
        'val': preds_val,
        'test': preds_test
    }

    try:
        with open(cached_prediction_filename, 'wb') as writefile:
            pickle.dump(preds, writefile)
    except:
        print('Error saving cached prediction file\n',
              cached_prediction_filename)

    return preds


def get_evaluations(
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_test,
    y_test,
    cached_evaluation_filename
):

    if os.path.exists(cached_evaluation_filename):
        with open(cached_evaluation_filename, 'rb') as readfile:
            print(f'\nUsing cached eval files {cached_evaluation_filename}')
            evaluations = pickle.load(readfile)

    else:
        train_eval = model.evaluate(X_train, y_train, verbose=1)
        print(f'\nTrain eval {X_train.shape} {train_eval}')
        print('\n')

        valid_eval = model.evaluate(X_valid, y_valid, verbose=1)
        print(f'\nValid eval {X_valid.shape} {valid_eval}')
        print('\n')

        test_eval = model.evaluate(X_test, y_test, verbose=1)
        print(f'\nTest eval {X_test.shape} {test_eval}')
        print('\n')

        evaluations = {
            'train': train_eval,
            'val': valid_eval,
            'test': test_eval
        }

        with open(cached_evaluation_filename, 'wb') as writefile:
            pickle.dump(evaluations, writefile)

    return evaluations


def get_predictions_thresholds(preds, preds_threshold=preds_threshold):
    preds_train = preds['train']
    preds_val = preds['val']
    preds_test = preds['test']

    preds_train_t = (preds_train > preds_threshold).astype(np.uint8)
    preds_val_t = (preds_val > preds_threshold).astype(np.uint8)
    preds_test_t = (preds_test > preds_threshold).astype(np.uint8)

    return preds_train_t, preds_val_t, preds_test_t


def plot_sample(model, X, y, preds, binary_preds, evaluations, ix=None):
    start = ix * square_dim
    end = start + square_dim

    x_image = gldata.stich_images(X[start:end]).astype(np.float32)

    y_image = gldata.stich_images(y[start:end]).astype(np.float32)

    preds_image = gldata.stich_images(preds[start:end]).astype(np.float32)

    binary_preds_image = gldata.stich_images(
        binary_preds[start:end]).astype(np.float32)

    has_mask = y_image.max() > 0

    plt.close()

    fig, ax = plt.subplots(
        1, 4, figsize=(40, 10)
    )

    metric_names = model.metrics_names
    loss_eval = evaluations[0]
    metric_eval = [
        f'{metric_name} {metric:0.3}' for metric, metric_name in list(zip(evaluations[1:], metric_names[1:]))]
    fig.suptitle(
        f'Satellite with Building Predicted {metric_names[0]}: {loss_eval:0.3f}, metric(s): {", ".join(metric_eval)})')

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
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_test,
    y_test,
    preds,
    evaluations,
    preds_threshold,
    split_len,
    parameters,
):

    dirpath = os.path.join(assets, parameters)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    preds_train = preds['train']
    preds_val = preds['val']
    preds_test = preds['test']

    preds_train_t, preds_val_t, preds_test_t = get_predictions_thresholds(
        preds, preds_threshold)

    train_eval = evaluations['train']
    val_eval = evaluations['val']
    test_eval = evaluations['test']

    # Check if training data looks all right
    if USE_X_TRAIN:
        train_min = min(CHECK_IMAGES, int(y_train.shape[0] / (split_len ** 2)))
        print(f'outputing predicted train images {train_min}')
        for i in tqdm(range(train_min), total=train_min):
            fig_train = plot_sample(
                model, X_train, y_train, preds_train, preds_train_t, train_eval, ix=i)
            figure_id = X_ids_train[i]
            name = f'{dirpath}/xview2_train_predicted_fig_{i}_{figure_id}.png'
            savefig(fig_train, name)

    # Check if validation data looks all right
    if USE_X_VALID:
        valid_min = min(CHECK_IMAGES, int(y_valid.shape[0] / (split_len ** 2)))
        print(f'outputing predicted valid images {valid_min}')
        for i in tqdm(range(valid_min), total=valid_min):
            fig_val = plot_sample(
                model, X_valid, y_valid, preds_val, preds_val_t, val_eval, ix=i)
            figure_id = X_ids_valid[i]
            name = f'{dirpath}/xview2_val_predicted_fig_{i}_{figure_id}.png'
            savefig(fig_val, name)

    # Check tests
    if USE_X_TEST:
        test_min = int(y_test.shape[0] / square_dim)
        print(f'outputing predicted test images {test_min}')
        for i in tqdm(range(test_min), total=test_min):
            fig_val = plot_sample(
                model, X_test, y_test, preds_test, preds_test_t, test_eval, ix=i)
            figure_id = ids_test[i]
            name = f'{dirpath}/xview2_test_predicted_fig_{i}_{figure_id}.png'
            savefig(fig_val, name)

    print('Finishing output')


def main_fast(X_train, y_train, X_valid, y_valid, X_test, y_test, train_percentage):

    total_time_start = time.time()

    # single
    batch_size = 64  # 32
    dropout = 0.1  # 0.1
    n_filters = 16  # 16
    epochs = 80  # 100
    patience = 15  # 20
    float_type = 'float16'

    # super fast
    # batch_size = 64  # 32
    # dropout = 0.1  # 0.1
    # n_filters = 4  # 16
    # epochs = 20  # 100
    # patience = 5  # 20
    # float_type = 'float16'

    # metrics
    # metrics = ['accuracy']  # ['accuracy']
    # metric_params = 'accuracy'  # 'accuracy'
    # loss_metrics = 'binary_crossentropy'  # 'binary_crossentropy'
    # loss_params = 'binary_crossentropy'  # 'binary_crossentropy'

    # alternate metrics
    metrics = ['accuracy', glunet.iou_coef,
               glunet.dice_coef_smooth]  # ['accuracy']
    metric_params = 'accuracy-iou-dice-smooth'  # 'accuracy'
    loss_metrics = glunet.iou_coef_loss  # 'binary_crossentropy'
    loss_params = 'iou-coef-loss'  # 'binary_crossentropy'

    # begin training
    model_params = []

    read_files_start = time.time()

    read_files_end = time.time() - read_files_start

    train_model_start = time.time()

    parameters = gldata.get_parameters(
        epochs,
        train_percentage,
        batch_size,
        dropout,
        n_filters,
        metric_params,
        loss_params,
        float_type,
    )
    CACHED_MODEL_FILENAME = f'models_xview2/xview2{parameters}.h5'
    CACHED_PREDICTION_FILENAME = f'models_xview2/xview2{parameters}_preds.pkl'
    CACHED_EVALUATION_FILENAME = f'models_xview2/xview2{parameters}_evals.pkl'

    print(f'\n\nparameters:\n{parameters}\n\n')

    # check_input_images(X_train, y_train, parameters)

    model = create_model(
        split_dimension,
        image_channels,
        n_filters,
        dropout,
        metrics,
        loss_metrics
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

    evaluations = get_evaluations(
        model,
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        CACHED_EVALUATION_FILENAME
    )

    output_train_val_test_images(
        model,
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        preds,
        evaluations,
        preds_threshold,
        split_len,
        parameters
    )

    predict_output_image_end = time.time() - predict_output_image_start

    total_time_end = time.time() - total_time_start

    prefix = f'minutes;  Read: {read_files_end // 60 }, train: {train_model_end // 60}, predict/output: {predict_output_image_end // 60}'

    print(f'\n\nparameters:\n{parameters}\n')

    model_params.append(
        (total_time_end // 60, prefix, CACHED_PREDICTION_FILENAME))

    return model, model_params, preds, evaluations


if __name__ == '__main__':
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_tvt(
        X_ids_train, X_ids_valid, ids_test, float_type)

    model, model_params, preds, evaluations = main_fast(
        X_train, y_train, X_valid, y_valid, X_test, y_test, train_percentage)

    print('\n\n\nParameters:\n', model_params)
