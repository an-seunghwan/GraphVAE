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
        
        self.mean_layer = layers.Dense(self.params['latent_dim'], activation='linear',
                                       use_bias=False)
        self.logvar_layer = layers.Dense(self.params['latent_dim'], activation='linear',
                                        use_bias=False)
        
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
        Ahat = tf.matmul(z, tf.transpose(z, [0, 2, 1]))
        # Ahat = tf.reshape(tf.matmul(z, tf.transpose(z, [0, 2, 1])), (-1, self.params['keywords'] * self.params['keywords']))
        # assert Ahat.shape == (self.params["batch_size"], self.params['keywords'] * self.params['keywords'])
        assert Ahat.shape == (x.shape[0], self.params['keywords'], self.params['keywords'])
        # assert Ahat.shape == (x.shape[0], self.params['keywords'] * self.params['keywords'])
        
        return mean, logvar, z, Ahat
#%%
class SparseGraphVAE(K.models.Model):
    def __init__(self, params):
        super(SparseGraphVAE, self).__init__()
        self.params = params
        
        w_init = tf.random_normal_initializer()
        self.w_mean = tf.Variable(initial_value=w_init(shape=(self.params['keywords'], self.params['latent_dim']),
                                dtype='float32'), trainable=True)
        self.w_var = tf.Variable(initial_value=w_init(shape=(self.params['keywords'], self.params['latent_dim']), 
                                dtype='float32'), trainable=True)

    def call(self, x):
        # encoder
        mean = tf.sparse.sparse_dense_matmul(x, self.w_mean)
        logvar = tf.sparse.sparse_dense_matmul(x, self.w_var)
        
        epsilon = tf.random.normal((self.params['keywords'], self.params['latent_dim']))
        z = mean + tf.math.exp(logvar / 2) * epsilon 
        assert z.shape == (self.params['keywords'], self.params['latent_dim'])
        
        # decoder
        Ahat = tf.reshape(tf.matmul(z, tf.transpose(z, [1, 0])), (1, self.params['keywords'] * self.params['keywords']))
        assert Ahat.shape == (1, self.params['keywords'] * self.params['keywords'])
        
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
    # error = tf.reduce_mean(K.losses.binary_crossentropy(A, Ahat, from_logits=True))
    error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(A, Ahat))
    '''가중치 부여'''
    # error = tf.reduce_mean(tf.multiply((beta*A)+1, tf.nn.sigmoid_cross_entropy_with_logits(A, Ahat)))
    # error = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(A, Ahat, pos_weight=beta))
        
    # KL loss by closed form
    kl = tf.reduce_mean(
        tf.reduce_sum(0.5 * (tf.math.pow(mean, 2) - 1 + tf.math.exp(logvar) - logvar), axis=(1, 2))
        )
        
    return error + kl, error, kl
#%%