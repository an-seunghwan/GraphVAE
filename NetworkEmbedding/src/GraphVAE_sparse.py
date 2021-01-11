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
import random
from scipy import sparse
import os
os.chdir('/Users/anseunghwan/Documents/GitHub/textmining')

import Modules
#%%
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
#%%
PARAMS = {
    "batch_size": 1000,
    "keywords": 300,
    "latent_dim": 2,
    "sigma": 1,
    "epochs": 100, 
    "beta_final": 1, # variance of observation model
    # "kl_anneal_rate": 0.05,
    # "logistic_anneal": True,
    "learning_rate": 0.01,
}

with open("./result/keywords.txt", "r") as f:
    keywords = [w.strip('\n') for w in f.readlines()]
#%%
month = 1
di = np.diag_indices(PARAMS['keywords'])
filelist = sorted([f for f in os.listdir('/Users/anseunghwan/Documents/uos/textmining/data/{}월/'.format(month)) if f.endswith('.npz')])
testn = np.argmin(np.array([os.path.getsize('/Users/anseunghwan/Documents/uos/textmining/data/{}월/'.format(month) + '/' + f) for f in filelist]))

'''validation(test)'''
Atest_ = sparse.load_npz('/Users/anseunghwan/Documents/uos/textmining/data/{}월/'.format(month) + filelist[testn])
Atest = Atest_.toarray().reshape((-1, PARAMS['keywords'], PARAMS['keywords']))
Atest[:, di[0], di[1]] = 1

# degree matrix
I = np.eye(PARAMS['keywords'])
D = I[None, :, :] * np.sqrt(1 / (np.sum(Atest[:, di[0], di[1]], axis=-1) - 1))[:, None, None]
Atest_tilde = tf.cast(D @ Atest @ D, tf.float32)
# A = tf.reshape(tf.cast(A, tf.float32), (-1, PARAMS['keywords'] * PARAMS['keywords']))
Atest = Atest.reshape(-1, PARAMS['keywords'] * PARAMS['keywords'])

filelist.remove(filelist[testn])
#%%
coo_matrix = sparse.coo_matrix(A_tilde[0, :, :])

tf_coo_matrix = tf.SparseTensor(
    indices=np.array([coo_matrix.row, coo_matrix.col]).T,
    values=coo_matrix.data,
    dense_shape=coo_matrix.shape)

# mean, logvar, z, Ahat = model(tf_coo_matrix)

w_init = tf.random_normal_initializer()
w_mean = tf.Variable(initial_value=w_init(shape=(PARAMS['keywords'], PARAMS['latent_dim']),
                    dtype='float32'), trainable=True)
mean = tf.sparse.sparse_dense_matmul(tf_coo_matrix, w_mean)

for _ in tqdm(range(1000)):
    mean = tf.sparse.sparse_dense_matmul(tf_coo_matrix, w_mean)
    

#%%
model = Modules.SparseGraphVAE(PARAMS)
learning_rate = tf.Variable(PARAMS["learning_rate"], trainable=False, name="LR")
optimizer = tf.keras.optimizers.RMSprop(learning_rate)

elbo = []
bce_losses = []
kl_losses = []
for epoch in range(1, PARAMS["epochs"] + 1):
    # KL annealing
    # beta = Modules.kl_anneal(epoch, int(PARAMS["epochs"] / 3), PARAMS) * PARAMS['beta_final'] # KL에 강제로 가중치 for better reconstruction
    # beta = 1 + ((PARAMS['beta_final'] - 1) / int(PARAMS["epochs"] * (2 / 3))) * epoch # reverse annealing
    
    # if epoch > PARAMS['epochs'] * (2 / 3):
    #     beta = PARAMS['beta_final']
    
    random.shuffle(filelist) # permutation 
    for i in tqdm(range(len(filelist))): 
        '''adjacency matrix'''
        adj = sparse.load_npz('/Users/anseunghwan/Documents/uos/textmining/data/{}월/'.format(month) + filelist[i])
        
        loss_ = 0
        with tf.GradientTape() as tape:
            for j in range(adj.shape[0]):
                A = adj[j].toarray().reshape((PARAMS['keywords'], PARAMS['keywords']))
                A[di[0], di[1]] = 1 # diagonal element
            
                '''degree matrix'''
                D = I * np.sqrt(1 / (np.sum(A[di[0], di[1]], axis=-1) - 1))
                A_tilde = tf.cast(D @ A @ D, tf.float32)
                # A = tf.reshape(tf.cast(A, tf.float32), (-1, PARAMS['keywords'] * PARAMS['keywords']))
                A = A.reshape(-1, PARAMS['keywords'] * PARAMS['keywords'])
                
                coo_matrix = sparse.coo_matrix(A_tilde)

                tf_coo_matrix = tf.SparseTensor(indices=np.array([coo_matrix.row, coo_matrix.col]).T,
                                                values=coo_matrix.data,
                                                dense_shape=coo_matrix.shape)
                
                mean, logvar, z, Ahat = model(tf_coo_matrix)
                loss, bce, kl_loss = Modules.loss_function(Ahat, A, mean, logvar, PARAMS['beta_final'], PARAMS) 
                loss_ += loss
                
                bce_losses.append(-1 * bce.numpy())
                kl_losses.append(-1 * kl_loss.numpy())
                elbo.append(-1 * bce.numpy() - kl_loss.numpy())
            
            grad = tape.gradient(loss_, model.weights)
            optimizer.apply_gradients(zip(grad, model.weights))
    
    # change temperature and learning rate
    new_lr = Modules.get_learning_rate(epoch, PARAMS['learning_rate'], PARAMS)
    learning_rate.assign(new_lr)

    print('\n')
    print("Epoch:", epoch, ", TRAIN loss:", loss.numpy())
    print("BCE:", bce.numpy(), ", KL loss:", kl_loss.numpy(), ", beta:", PARAMS['beta_final'])
    
    mean, logvar, z, Ahat = model(Atest_tilde)
    loss, _, _ = Modules.loss_function(Ahat, Atest, mean, logvar, PARAMS['beta_final'], PARAMS) 
    print("Eval Loss:", loss.numpy()) 
    print('\n')
#%%
mean, logvar, z, Ahat = model(Atest_tilde)
#%%
'''
각 기사(n)에서 사용된 keyword들에 대해서만 z를 sampling하고 시각화
'''
for n in range(len(z)):
    zmat = np.array(z)
    idx = np.where(np.diag(Atest_.toarray()[n, :].reshape(PARAMS['keywords'], PARAMS['keywords'])) > 0)[0]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(zmat[n, idx, 0], zmat[n, idx, 1], s=10)
    for i in idx:
        ax.annotate(keywords[i], (zmat[n, i, 0], zmat[n, i, 1]), fontsize=10)
    plt.savefig('./result/{}월/sample{}.png'.format(month, n), 
                dpi=200, bbox_inches="tight", pad_inches=0.1)
#%%
'''
각 기사(n)에서 사용된 keyword들에 대해서만 z의 center를 시각화
'''
for n in range(len(z)):
    meanmat = np.array(mean)
    idx = np.where(np.diag(Atest_.toarray()[n, :].reshape(PARAMS['keywords'], PARAMS['keywords'])) > 0)[0]
    fig, ax = plt.subplots(figsize=(7, 7))
    # ax.set_xlim(np.min(meanmat[n, idx, 0]), np.max(meanmat[n, idx, 0]))
    # ax.set_ylim(np.min(meanmat[n, idx, 1]), np.max(meanmat[n, idx, 1]))
    ax.scatter(meanmat[n, idx, 0], meanmat[n, idx, 1], s=10)
    for i in idx:
        ax.annotate(keywords[i], (meanmat[n, i, 0], meanmat[n, i, 1]), fontsize=10)
    plt.savefig('./result/{}월/center{}.png'.format(month, n), 
                dpi=200, bbox_inches="tight", pad_inches=0.1)
#%%
n = 1
meanmat = np.array(mean)
idx = np.where(np.diag(Atest_.toarray()[n, :].reshape(PARAMS['keywords'], PARAMS['keywords'])) > 0)[0]
fig, ax = plt.subplots(figsize=(20, 20))
# ax.set_xlim(np.min(meanmat[n, idx, 0]), np.max(meanmat[n, idx, 0]))
# ax.set_ylim(np.min(meanmat[n, idx, 1]), np.max(meanmat[n, idx, 1]))
ax.scatter(meanmat[n, :, 0], meanmat[n, :, 1], s=10)
for i in range(len(keywords)):
    ax.annotate(keywords[i], (meanmat[n, i, 0], meanmat[n, i, 1]), fontsize=10)
#%%
# reconstruction
# def sigmoid(z):
#     return 1/(1 + np.exp(-z))
# reconA = np.array(sigmoid(zmat[n, :, :] @ zmat[n, :, :].T) > 0.5, dtype=int)
#%%
# learning phase
plt.rc('xtick', labelsize=10)   
plt.rc('ytick', labelsize=10)   
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(bce_losses, color='black', label='neg BCE')
ax.plot(kl_losses, color='darkorange', label='neg KL')
ax.plot(elbo, color='red', label='ELBO')
leg = ax.legend(fontsize=15, loc='lower right')
plt.savefig('./result/{}월/learning_phase.png'.format(month), 
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%