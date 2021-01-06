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

import Modules
#%%
PARAMS = {
    "batch_size": 1000,
    "keywords": 500,
    "latent_dim": 2,
    "sigma": 1,
    "epochs": 100, 
    "beta_final": 1, # variance of observation model
    "kl_anneal_rate": 0.05,
    "logistic_anneal": True,
    "learning_rate": 0.005,
}
#%%
month = 1
di = np.diag_indices(PARAMS['keywords'])
filelist = [f for f in os.listdir('/Users/anseunghwan/Documents/uos/textmining/data/{}월/'.format(month)) if f.endswith('.npz')]
I = np.eye(PARAMS['keywords'])
#%%
model = Modules.GraphVAE(PARAMS)
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
    
    for i in tqdm(range(len(filelist))):
        A = sparse.load_npz('/Users/anseunghwan/Documents/uos/textmining/data/{}월/'.format(month) + filelist[i])
        A = A.toarray().reshape((-1, PARAMS['keywords'], PARAMS['keywords']))

        '''degree matrix'''
        D = I[None, :, :] * np.sqrt(1 / (np.sum(A[:, di[0], di[1]], axis=-1) - 1))[:, None, None]
        A_tilde = tf.cast(D @ A @ D, tf.float32)
        A = tf.reshape(tf.cast(A, tf.float32), (-1, PARAMS['keywords'] * PARAMS['keywords']))
        
        with tf.GradientTape(persistent=True) as tape:
            mean, logvar, z, Ahat = model(A_tilde)
            loss, bce, kl_loss = Modules.loss_function(Ahat, A, mean, logvar, PARAMS['beta_final'], PARAMS) 
            
            bce_losses.append(-1 * bce.numpy())
            kl_losses.append(-1 * kl_loss.numpy())
            elbo.append(-1 * bce.numpy() - kl_loss.numpy())
            
        grad = tape.gradient(loss, model.weights)
        optimizer.apply_gradients(zip(grad, model.weights))
    
    # change temperature and learning rate
    new_lr = Modules.get_learning_rate(epoch, PARAMS['learning_rate'], PARAMS)
    learning_rate.assign(new_lr)

    print('\n')
    print("Epoch:", epoch, ", TRAIN loss:", loss.numpy())
    print("BCE:", bce.numpy(), ", KL loss:", kl_loss.numpy(), ", beta:", PARAMS['beta_final'])
    
    # visualization
    # if epoch % 10 == 0:
    #     GMGS_module.center_reconstruction(model, epoch, PARAMS)
    #     GMGS_module.example_reconstruction(train_x, y, xhat)
    #     GMGS_module.z_space(z, PARAMS)
    
    # if epoch % 1 == 0:
    #     losses = [] # only ELBO loss
    #     for test_x, test_y in test_dataset:
    #         mean, logvar, logits, y, z, z_tilde, xhat = model(test_x, tau)
    #         if key == 'proposal':
    #             loss, _, _ = GMGS_module.loss_closedform(logits, xhat, test_x, mean, logvar, tau, beta, PARAMS) 
    #         else:
    #             loss, _, _ = GMGS_module.loss_without(logits, xhat, test_x, mean, logvar, tau, beta, PARAMS) 
    #         loss2 = sce_loss(test_y, logits)
    #         # cross entropy weight 조절 (annealing)
    #         # loss = loss + 10 * loss2 # annealing on cross-entropy beta = 1
    #         # loss = loss + 5 * loss2 # annealing on cross-entropy beta = 0.5
    #         loss = loss + loss2 # annealing on cross-entropy beta = 0.05
    #         losses.append(loss)
    #     eval_loss = np.mean(losses)
    #     print("Eval Loss:", eval_loss, "\n") 
#%%