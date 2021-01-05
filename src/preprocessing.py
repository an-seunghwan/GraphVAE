#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print('==========================================')
print(device_lib.list_local_devices())
tf.debugging.set_log_device_placement(False)
#%%
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
import time
import re
import matplotlib.pyplot as plt
from PIL import Image
from pprint import pprint
import os
os.chdir('/Users/anseunghwan/Documents/GitHub/textmining')
#%%
PARAMS = {
    "batch_size": 2000,
    "data_dim": 784,
    "class_num": 10,
    "latent_dim": 2,
    "sigma": 4,
    "epochs": 150, 
    "beta_final": 0.002, # variance of observation model
    "kl_anneal_rate": 0.05,
    "logistic_anneal": True,
    "temperature_anneal_rate": 0.0003,
    "init_temperature": 3.0, 
    "min_temperature": 1.0,
    "learning_rate": 0.005,
    "hard": True,
    "FashionMNIST": False,
}
#%%
data = pd.read_csv('/Users/anseunghwan/Documents/uos/textmining/한국보건사회연구원_데이터_형태소_1월_200818.csv',
                   encoding='cp949')
#%%
data.head()
data.shape
data.columns
keywords = list(data.columns[9:])

#%%