# Takes 7 minutes on 16 core to split images (each image splits to 25 smaller images)
# 4000 (640, 640, 3) --> 100000 (128, 128, 3)

import os
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook, tnrange, tqdm

from src.utilities import data as gldata

split_dimension = 128
image_dimension = 640
image_channels = 3
mask_channels = 1

train_percentage = 1
# train_percentage = 0.01

ids = gldata.get_image_ids()

if train_percentage != 1:
    train_index = int((len(ids) * train_percentage) // 1)
    ids = ids[:train_index]

print(f'Splitting {len(ids)} images')

gldata.create_data(
    ids,
    split_dimension,
    image_dimension,
    image_channels,
    mask_channels
)
