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
di = np.diag_indices(PARAMS['keywords'])
A = sparse.load_npz('./results/A.npz')
A = A.toarray().reshape((-1, PARAMS['keywords'], PARAMS['keywords']))

'''degree matrix'''
I = np.eye(PARAMS['keywords'])
D = I * np.sqrt(1/(sum((A)) - 1))
D = I[None, :, :] * np.sqrt(1 / (np.sum(A[:, di[0], di[1]], axis=-1) - 1))[:, None, None]
A_tilde = D @ A @ D

A_tilde = tf.cast(A_tilde, tf.float32)
#%%
model = Modules.GraphVAE(PARAMS)
learning_rate = tf.Variable(PARAMS["learning_rate"], trainable=False, name="LR")
optimizer = tf.keras.optimizers.RMSprop(learning_rate)

mean, logvar, z, Ahat = model(A_tilde)
A = tf.reshape(tf.cast(A, tf.float32), (-1, PARAMS['keywords'] * PARAMS['keywords']))

loss, bce, kl_loss = Modules.loss_function(Ahat, A, mean, logvar, PARAMS['beta_final'], PARAMS) 

elbo = []
bce_losses = []
kl_losses = []
for epoch in range(1, PARAMS["epochs"] + 1):
    
    # KL annealing
    # beta = Modules.kl_anneal(epoch, int(PARAMS["epochs"] / 3), PARAMS) * PARAMS['beta_final'] # KL에 강제로 가중치 for better reconstruction
    # beta = 1 + ((PARAMS['beta_final'] - 1) / int(PARAMS["epochs"] * (2 / 3))) * epoch # reverse annealing
    
    # if epoch > PARAMS['epochs'] * (2 / 3):
    #     beta = PARAMS['beta_final']
    
    for train_x, train_y in train_dataset:
        with tf.GradientTape(persistent=True) as tape:
            mean, logvar, logits, y, z, z_tilde, xhat = model(train_x, tau)
            loss, mse, kl_loss = Modules.loss_mixture(logits, xhat, train_x, mean, logvar, tau, PARAMS['beta_final'], PARAMS) 
            # loss, mse, kl_loss = Modules.loss_mixture(logits, xhat, train_x, mean, logvar, tau, beta, PARAMS) 
            sce = sce_loss(train_y, logits)
            loss_ = loss + alpha * PARAMS['beta_final'] * sce # cross entropy weight 조절 ?
            # loss_ = loss + alpha * beta * sce 
            
            mse_losses.append(-1 * (mse.numpy() / PARAMS['beta_final']))
            kl_losses.append(-1 * kl_loss.numpy())
            sce_losses.append(-1 * sce.numpy())
            elbo.append(-1 * (mse.numpy() / PARAMS['beta_final']) - 
                        kl_loss.numpy() - 
                        sce.numpy() - 
                        (PARAMS['latent_dim'] / 2) * np.log(2 * np.pi * PARAMS['beta_final']))
            
        grad = tape.gradient(loss_, model.weights)
        optimizer.apply_gradients(zip(grad, model.weights))
    
    # change temperature and learning rate
    new_lr = Modules.get_learning_rate(epoch, PARAMS['learning_rate'], PARAMS)
    learning_rate.assign(new_lr)

    print("Epoch:", epoch, ", TRAIN loss:", loss_.numpy(), ", Temperature:", tau)
    print("MSE:", mse.numpy(), ", CCE:", sce.numpy() ,", KL loss:", kl_loss.numpy(), ", beta:", PARAMS['beta_final'])
    # print("MSE:", mse.numpy(), ", CCE:", sce.numpy() ,", KL loss:", kl_loss.numpy(), ", beta:", beta)
    print(np.round(logits.numpy()[0], 3))
    
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