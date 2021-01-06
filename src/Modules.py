#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
import numpy as np
import math
import time
import re
import matplotlib.pyplot as plt
#%%
class GraphVAE(K.models.Model):
    def __init__(self, params):
        super(GraphVAE, self).__init__()
        self.params = params
        
        self.mean_layer = layers.Dense(self.params['latent_dim'], activation='linear')
        self.logvar_layer = layers.Dense(self.params['latent_dim'], activation='linear')

    def call(self, x):
        # encoder
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        
        # epsilon = tf.random.normal((self.params["batch_size"], self.params['keywords'], self.params['latent_dim']))
        epsilon = tf.random.normal((x.shape[0], self.params['keywords'], self.params['latent_dim']))
        z = mean + tf.math.exp(logvar / 2) * epsilon 
        # assert z.shape == (self.params["batch_size"], self.params['keywords'], self.params['latent_dim'])
        assert z.shape == (x.shape[0], self.params['keywords'], self.params['latent_dim'])
        
        # decoder
        Ahat = tf.reshape(tf.matmul(z, tf.transpose(z, [0, 2, 1])), (-1, self.params['keywords'] * self.params['keywords']))
        # assert Ahat.shape == (self.params["batch_size"], self.params['keywords'] * self.params['keywords'])
        assert Ahat.shape == (x.shape[0], self.params['keywords'] * self.params['keywords'])
        
        return mean, logvar, z, Ahat
#%%
def kl_anneal(epoch, epochs, PARAMS):
    if PARAMS['logistic_anneal']:
        return 1 / (1 + math.exp(-PARAMS['kl_anneal_rate']*(epoch - epochs)))
    else:
        return (1 / (epochs * 2)) * epoch
#%%
def get_learning_rate(epoch, init, PARAMS):
    return tf.convert_to_tensor(init * pow(0.95, (epoch / PARAMS["epochs"])), dtype=tf.float32)
#%%
def loss_function(Ahat, A, mean, logvar, beta, PARAMS):
    # reconstruction
    error = tf.reduce_mean(K.losses.binary_crossentropy(A, Ahat, from_logits=True))
    
    temp = np.isnan(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.sigmoid(Ahat), labels=A).numpy())
    tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.sigmoid(Ahat), labels=A).numpy()[temp]
    Ahat.numpy()[temp]
    
    np.where(np.isnan(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.sigmoid(Ahat), labels=A).numpy()))[0][0]
    Ahat.numpy()[158, :]
    
    np.unique(A.numpy()[158, :])
   
    z.numpy()[158, :]
    
    A_tilde.numpy()[158, :]
    
    mean.numpy()[158, :]
    logvar.numpy()[158, :]
    
    tf.sigmoid(Ahat).numpy()[35, :]
    A[35: ]
    
    # KL loss by closed form
    kl = tf.reduce_mean(
        tf.reduce_sum(0.5 * (tf.math.pow(mean, 2) - 1 + tf.math.exp(logvar) - logvar), axis=(1, 2))
        )
        
    return error + beta * kl, error, kl
#%%