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
from scipy import sparse
import os
os.chdir('/Users/anseunghwan/Documents/GitHub/textmining')

import Module
#%%
PARAMS = {
    "batch_size": 1000,
    # "data_dim": 784,
    # "class_num": 10,
    "latent_dim": 2,
    "sigma": 1,
    "epochs": 150, 
    "beta_final": 0.01, # variance of observation model
    "kl_anneal_rate": 0.05,
    "logistic_anneal": True,
    # "temperature_anneal_rate": 0.0003,
    # "init_temperature": 3.0, 
    # "min_temperature": 1.0,
    "learning_rate": 0.005,
    # "hard": True,
}
#%%
data = pd.read_csv('/Users/anseunghwan/Documents/uos/textmining/한국보건사회연구원_데이터_형태소_1월_200818.csv',
                   encoding='cp949').iloc[:1000]
print(data.head())
print(data.shape)
keywords = list(data.columns[9:])
PARAMS['keywords'] = len(keywords)
#%%
'''frequency 정보를 모두 1로 맞춤'''
freq = np.array(data[data.columns[9:]])
freq[np.where(freq > 0)] = 1
#%%
# '''adjacency matrix'''
# adj = tf.matmul(freq[:, :, None], freq[:, None, :]).numpy()
# di = np.diag_indices(len(keywords))
# adj[:, di] = 1
# np.sqrt(1/(freq.sum(axis=1) - 1))
#%%
adj = []
# adj = np.zeros((1, len(keywords), len(keywords)))
di = np.diag_indices(len(keywords))
I = np.eye(len(keywords))
for i in tqdm(range(len(freq))):
    '''adjacency matrix'''
    A = np.array(sparse.csr_matrix(freq[[i], :].T).multiply(sparse.csr_matrix(freq[[i], :])).toarray())
    A[di] = 1
    '''degree matrix'''
    D = I * np.sqrt(1/(sum(freq[i, :]) - 1))
    A = D @ A @ D
    # adj = np.concatenate((adj, A[None, :, :]), axis=0)
    adj.append(A)
adj = np.array(adj)
adj = sparse.csr_matrix(adj)
sparse.save_npz('./result/sparse_matrix.npz', adj)
#%%
x = tf.cast(adj, tf.float32)
#%%

#%%

#%%