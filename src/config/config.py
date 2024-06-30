import pathlib
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop

import src

# Create the training dataset
training_data = None
X_train = None
Y_train = None

epochs = 100
mb_size = 2

PACKAGE_ROOT = pathlib.Path(src.__file__).resolve().parent
DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")

SAVED_MODEL_PATH = os.path.join(PACKAGE_ROOT,"trained_models")