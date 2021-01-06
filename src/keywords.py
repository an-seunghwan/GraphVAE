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
#%%
'''keyword의 frequency가 1인 데이터는 제거'''
data = []
for i in tqdm(range(10)):
    data.append(pd.read_csv('/Users/anseunghwan/Documents/uos/textmining/한국보건사회연구원_데이터_형태소_{}월.csv'.format(i+1),
                encoding='cp949'))
#%%
'''keyword selection'''
keyidx = np.argsort(np.array(data[data.columns[9:]].sum(axis=0)))[-100:]
keywords = list(data.columns[9:][keyidx])
PARAMS['keywords'] = len(keywords)
# data[data.columns[9:]].sum(axis=0)[np.argsort(np.array(data[data.columns[9:]].sum(axis=0)))[-100:]]
#%%
'''frequency 정보를 모두 1로 맞춤'''
freq = np.array(data[data.columns[9:][keyidx]])
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
for i in tqdm(range(len(freq))):
    '''adjacency matrix'''
    A = np.array(sparse.csr_matrix(freq[[i], :].T).multiply(sparse.csr_matrix(freq[[i], :])).toarray())
    # adj = np.concatenate((adj, A[None, :, :]), axis=0)
    adj.append(A)
adj = np.array(adj).reshape(len(adj), -1)
adj = sparse.csr_matrix(adj)
sparse.save_npz('./results/A.npz', adj)