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
    "epochs": 2, 
    "beta": 1, 
    # "kl_anneal_rate": 0.05,
    # "logistic_anneal": True,
    "learning_rate": 0.05,
}

with open("./result/keywords.txt", "r") as f:
    keywords = [w.strip('\n') for w in f.readlines()]
#%%
di = np.diag_indices(PARAMS['keywords'])
filelist = sorted([f for f in os.listdir('/Users/anseunghwan/Documents/uos/textmining/data/') if f.endswith('.npz')])
# testn = np.argmin(np.array([os.path.getsize('/Users/anseunghwan/Documents/uos/textmining/data/' + f) for f in filelist]))
testn = len(filelist)-1

I = np.eye(PARAMS['keywords'])
'''validation(test)'''
Atest_ = sparse.load_npz('/Users/anseunghwan/Documents/uos/textmining/data/Atest.npz')
Atest = Atest_.toarray().reshape((-1, PARAMS['keywords'], PARAMS['keywords']))

'''degree matrix (every node connected to itself)'''
# D = I[None, :, :] * np.sqrt(1 / np.sum(Atest_[:, di[0], di[1]], axis=-1))[:, None, None]
D = Atest[:, di[0], di[1]] * np.sqrt(1 / np.sum(Atest[:, di[0], di[1]], axis=-1, keepdims=True))
D[np.where(D == 0)] = 1
D = I[None, :, :] * D[:, None]
Atest[:, di[0], di[1]] = 1 # diagonal element

'''matmul with tensorflow (faster)'''
# Atest_tilde = tf.cast(D @ Atest @ D, tf.float32)
Atest = tf.cast(Atest, tf.float32)
Atest_tilde = tf.matmul(tf.matmul(tf.cast(D, tf.float32), Atest), tf.cast(D, tf.float32))

'''reshape'''
# A = tf.reshape(tf.cast(A, tf.float32), (-1, PARAMS['keywords'] * PARAMS['keywords']))
# Atest = tf.reshape(tf.cast(Atest, tf.float32), (-1, PARAMS['keywords'] * PARAMS['keywords']))
# Atest = Atest.reshape(-1, PARAMS['keywords'] * PARAMS['keywords'])

filelist.remove(filelist[testn])
#%%
model = Modules.GraphVAE(PARAMS)
# learning_rate = tf.Variable(PARAMS["learning_rate"], trainable=False, name="LR")
optimizer = K.optimizers.Adam(PARAMS["learning_rate"])
# bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

I = tf.eye(PARAMS['keywords'], PARAMS['keywords'])

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
        A = sparse.load_npz('/Users/anseunghwan/Documents/uos/textmining/data/' + filelist[i])
        A = A.toarray().reshape((-1, PARAMS['keywords'], PARAMS['keywords']))
        
        '''degree matrix'''
        # D = I[None, :, :] * np.sqrt(1 / (np.sum(A[:, di[0], di[1]], axis=-1) - 1))[:, None, None]
        D = A[:, di[0], di[1]] * np.sqrt(1 / np.sum(A[:, di[0], di[1]], axis=-1, keepdims=True))
        D[np.where(D == 0)] = 1
        D = tf.multiply(I[tf.newaxis, :, :], tf.cast(D, tf.float32)[:, tf.newaxis])
        # D = I[None, :, :] * D[:, None]
        A[:, di[0], di[1]] = 1 # diagonal element set 1
        
        '''input normalized adjacency'''
        '''matmul with tensorflow (faster)'''
        # A_tilde = tf.cast(D @ A @ D, tf.float32)
        # D = tf.cast(D, tf.float32)
        A = tf.cast(A, tf.float32)
        A_tilde = tf.matmul(tf.matmul(D, A), D)
        
        '''reshape'''
        # A = tf.reshape(tf.cast(A, tf.float32), (-1, PARAMS['keywords'] * PARAMS['keywords']))
        # A = A.reshape(-1, PARAMS['keywords'] * PARAMS['keywords'])
                
        with tf.GradientTape(persistent=False) as tape:
            mean, logvar, z, Ahat = model(A_tilde)
            loss, bce, kl_loss = Modules.loss_function(Ahat, A, mean, logvar, PARAMS['beta'], PARAMS) 
            
            bce_losses.append(-1 * bce.numpy())
            kl_losses.append(-1 * kl_loss.numpy())
            elbo.append(-1 * bce.numpy() - kl_loss.numpy())
            
        grad = tape.gradient(loss, model.weights)
        optimizer.apply_gradients(zip(grad, model.weights))
    
    # change temperature and learning rate
    # new_lr = Modules.get_learning_rate(epoch, PARAMS['learning_rate'], PARAMS)
    # learning_rate.assign(new_lr)

    print('\n')
    print("Epoch:", epoch, ", TRAIN loss:", loss.numpy())
    print("BCE:", bce.numpy(), ", KL loss:", kl_loss.numpy(), ", beta:", PARAMS['beta'])
    
    # test
    mean, logvar, z, Ahat = model(Atest_tilde)
    loss, _, _ = Modules.loss_function(Ahat, Atest, mean, logvar, PARAMS['beta'], PARAMS) 
    print("Eval Loss:", loss.numpy()) 
    print('\n')
#%%
'''model save'''
np.save('./result/mean_weight', model.weights[0].numpy())
np.save('./result/logvar_weight', model.weights[1].numpy())

# meanmat = np.array(mean)
# idx = np.where(Atest_.toarray().reshape(-1, PARAMS['keywords'], PARAMS['keywords'])[:, di[0], di[1]] > 0)
# meanmat = np.unique(meanmat[idx[0], idx[1], :], axis=0)
# plt.figure(figsize=(10, 10))
# plt.rc('xtick', labelsize=10)   
# plt.rc('ytick', labelsize=10)   
# plt.scatter(meanmat[:, 0], meanmat[:, 1], c=sum([[i]*100 for i in range(10)], []), s=15, cmap=plt.cm.Reds, alpha=1)
# plt.savefig('./result/clustering2.png', 
#             dpi=200, bbox_inches="tight", pad_inches=0.1)

mean, logvar, z, Ahat = model(Atest_tilde)
#%%
'''load model'''
# wmean = np.load('./result/210108/mean_weight.npy')
# wlogvar = np.load('./result/210108/logvar_weight.npy')
# mean_layer2 = layers.Dense(PARAMS['latent_dim'], activation='linear',
#                             use_bias=False)
# logvar_layer2 = layers.Dense(PARAMS['latent_dim'], activation='linear',
#                             use_bias=False)
# mean_layer2(tf.ones((1, 300, 300)))
# logvar_layer2(tf.ones((1, 300, 300)))
# mean_layer2.set_weights([wmean])
# logvar_layer2.set_weights([wlogvar])

# # model
# input_layer = layers.Input((PARAMS['keywords'], PARAMS['keywords']))
# mean_ = mean_layer2(input_layer)
# logvar_ = logvar_layer2(input_layer)
# epsilon = tf.random.normal((PARAMS['keywords'], PARAMS['latent_dim']))
# z_ = mean_ + tf.math.exp(logvar_ / 2) * epsilon 
# Ahat_ = tf.matmul(z_, tf.transpose(z_, [0, 2, 1]))
# model2 = K.models.Model(input_layer, [mean_, logvar_, z_, Ahat_])

# mean, logvar, z, Ahat = model2(Atest_tilde)
#%%
'''
각 기사(n)에서 사용된 keyword들에 대해서만 z를 sampling
'''
# for n in range(len(z)):
#     zmat = np.array(z)
#     idx = np.where(np.diag(Atest_.toarray()[n, :].reshape(PARAMS['keywords'], PARAMS['keywords'])) > 0)[0]
#     fig, ax = plt.subplots(figsize=(7, 7))
#     ax.scatter(zmat[n, idx, 0], zmat[n, idx, 1], s=10)
#     for i in idx:
#         ax.annotate(keywords[i], (zmat[n, i, 0], zmat[n, i, 1]), fontsize=10)
#     plt.savefig('./result/{}월/sample{}.png'.format(month, n), 
#                 dpi=200, bbox_inches="tight", pad_inches=0.1)
#%%
'''
각 기사(n)에서 사용된 keyword들에 대해서 z의 center
'''
# for n in range(len(z)):
#     meanmat = np.array(mean)
#     idx = np.where(np.diag(Atest_.toarray()[n, :].reshape(PARAMS['keywords'], PARAMS['keywords'])) > 0)[0]
#     fig, ax = plt.subplots(figsize=(7, 7))
#     # ax.set_xlim(np.min(meanmat[n, idx, 0]), np.max(meanmat[n, idx, 0]))
#     # ax.set_ylim(np.min(meanmat[n, idx, 1]), np.max(meanmat[n, idx, 1]))
#     ax.scatter(meanmat[n, idx, 0], meanmat[n, idx, 1], s=10)
#     for i in idx:
#         ax.annotate(keywords[i], (meanmat[n, i, 0], meanmat[n, i, 1]), fontsize=10)
#     plt.savefig('./result/center{}.png'.format(n), 
#                 dpi=200, bbox_inches="tight", pad_inches=0.1)

'''
article의 대표벡터
'''
meanmat = np.array(mean)
# idx = np.where(Atest_.toarray().reshape(-1, PARAMS['keywords'], PARAMS['keywords'])[:, di[0], di[1]] > 0)
# meanmat = np.unique(meanmat[idx[0], idx[1], :], axis=0)
article = []
for n in tqdm(range(1000)):
    idx = np.where(Atest_.toarray()[n, :].reshape(PARAMS['keywords'], PARAMS['keywords'])[di[0], di[1]] > 0)
    article.extend(np.unique(meanmat[n, idx[0], :], axis=0))
article = np.array(article)
plt.figure(figsize=(15, 15))
plt.xticks(size = 25)
plt.yticks(size = 25)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))  
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0)) 
plt.scatter(article[:, 0], article[:, 1], c=sum([[i]*100 for i in range(10)[::-1]], []), s=40, cmap=plt.cm.Reds, alpha=1)
plt.scatter(0, 0, color='darkred', marker='D')
plt.savefig('./result/clustering.png', 
            dpi=200, bbox_inches="tight", pad_inches=0.1)
#%%
'''
월별 대표벡터
'''
meanmat = np.array(mean)
for k in range(10):
    article = []
    for n in tqdm(range(100*k, 100*(k+1))):
        idx = np.where(Atest_.toarray()[n, :].reshape(PARAMS['keywords'], PARAMS['keywords'])[di[0], di[1]] > 0)
        article.extend(np.unique(meanmat[n, idx[0], :], axis=0))
    article = np.array(article)
    plt.figure(figsize=(15, 15))
    plt.xticks(size = 20)
    plt.yticks(size = 20)   
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))  
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0)) 
    # plt.xlim((np.min(meanmat[:, 0]), np.max(meanmat[:, 0])))
    # plt.ylim((np.min(meanmat[:, 1]), np.max(meanmat[:, 1])))
    # plt.title('{}월'.format(k+1)) 
    plt.scatter(article[:, 0], article[:, 1], s=70)
    plt.scatter(0, 0, color='darkred', marker='D', s=70)
    plt.savefig('./result/clustering_{}.png'.format(k), 
                dpi=200, bbox_inches="tight", pad_inches=0.1)

# for n in range(len(z)):
    # meanmat = np.array(mean)
    # fig, ax = plt.subplots(figsize=(10, 10))
    # # ax.set_xlim(np.min(meanmat[n, idx, 0]), np.max(meanmat[n, idx, 0]))
    # # ax.set_ylim(np.min(meanmat[n, idx, 1]), np.max(meanmat[n, idx, 1]))
    # ax.scatter(meanmat[n, :, 0], meanmat[n, :, 1], s=10)
    # for i in range(len(keywords)):
    #     ax.annotate(keywords[i], (meanmat[n, i, 0], meanmat[n, i, 1]), fontsize=10)
    # # plt.savefig('./result/center{}.png'.format(n), 
    # #             dpi=200, bbox_inches="tight", pad_inches=0.1)
#%%
'''
가장 거리가 먼 기사
'''
meanmat = np.array(mean)
for k in range(10):
    print(k)
    article = []
    for n in range(100*k, 100*(k+1)):
        idx = np.where(Atest_.toarray()[n, :].reshape(PARAMS['keywords'], PARAMS['keywords'])[di[0], di[1]] > 0)
        article.extend(np.unique(meanmat[n, idx[0], :], axis=0))
    article = np.array(article)
    dist = np.linalg.norm(article - article[:, None], axis=-1)
    distargs = np.sort(dist.reshape(-1, ))[-10:]
    for j in range(10):
        idx_ = np.where(dist == distargs[j])[0]
        # idx_ = np.where(dist == np.max(dist))[0]
        for u in idx_:
            print(', '.join([keywords[i] for i in np.where(np.diag(Atest_.toarray()[100*k + u, :].reshape(300, 300)) == 1)[0]]))
            # print(', '.join([keywords[i] for i in np.where(np.diag(Atest_.toarray()[100*k + idx_[1], :].reshape(300, 300)) == 1)[0]]))
        print('-------------------------')
#%%
'''embedding vector'''
emb = model.weights[0].numpy()
fig, ax = plt.subplots(figsize=(25, 25))
# ax.set_xlim(np.min(meanmat[n, idx, 0]), np.max(meanmat[n, idx, 0]))
# ax.set_ylim(np.min(meanmat[n, idx, 1]), np.max(meanmat[n, idx, 1]))
ax.scatter(emb[:, 0], emb[:, 1], s=20)
plt.xticks(size = 30)
plt.yticks(size = 30)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))  
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0)) 
for i in range(len(keywords)):
    ax.annotate(keywords[i], (emb[i, 0], emb[i, 1]), fontsize=20)
plt.savefig('./result/emb.png', 
            dpi=200, bbox_inches="tight", pad_inches=0.1)
#%%
'''test data 분석'''
Atest_ = Atest_.toarray().reshape(-1, PARAMS['keywords'], PARAMS['keywords'])[:, di[0], di[1]]

test_words = []
for i in range(len(Atest_)):
    test_words.append([keywords[w] for w in np.where(Atest_[i, :] == 1)[0]])
    
# test words save
with open("./result/test_words.txt", "w") as f:
    for w in test_words:
        f.write(' '.join(w) + '\n')

plt.figure(figsize=(20, 20))
for j in range(10):
    plt.subplot(5, 2, j+1)
    count = {x:0 for x in keywords}
    temp = sum(test_words[100*j:100*(j+1)], [])
    for i in range(len(temp)):
        count[temp[i]] = count.get(temp[i]) + 1
    plt.bar(range(len(count)), list(count.values()), align='center')
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.title('{}월'.format(j+1), fontsize=30)
    # plt.xticks(range(len(count)), list(count.keys()))
    plt.tight_layout() 
plt.savefig('./result/test_freq.png', 
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
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
plt.savefig('./result/learning_phase.png', 
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%