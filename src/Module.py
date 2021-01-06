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
        
        epsilon = tf.random.normal((self.params["batch_size"], self.params['keywords'], self.params['latent_dim']))
        z = mean + tf.math.exp(logvar / 2) * epsilon 
        assert z.shape == (self.params["batch_size"], self.params['keywords'], self.params['latent_dim'])
        
        # decoder
        Ahat = tf.matmul(z, tf.transpose(z, [0, 2, 1]))
        assert Ahat.shape == (self.params["batch_size"], self.params['keywords'], self.params['keywords'])
        
        return mean, logvar, z, Ahat
#%%
class MixtureVAE(K.models.Model):
    def __init__(self, params):
        super(MixtureVAE, self).__init__()
        self.params = params
        
        '''
        leaky relu is better than tanh where latent variable's value is large
        '''
        
        # encoder
        self.enc_dense1 = layers.Dense(256, activation='linear')
        
        # continuous latent
        self.mean_layer = [layers.Dense(self.params['latent_dim'], activation='linear') for _ in range(self.params['class_num'])]
        self.logvar_layer = [layers.Dense(self.params['latent_dim'], activation='linear') for _ in range(self.params['class_num'])]
        
        # discrete latent
        self.logits = layers.Dense(self.params["class_num"], activation='softmax') # non-negative logits

        # decoder
        self.dec_dense1 = layers.Dense(256, activation='linear')
        self.dec_dense2 = layers.Dense(self.params["data_dim"], activation='sigmoid')
        
    def sample_gumbel(self, shape, eps=1e-20): 
        '''Sampling from Gumbel(0, 1)'''
        U = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature): 
        '''
        Draw a sample from the Gumbel-Softmax distribution
        - logits: unnormalized
        - temperature: non-negative scalar (annealed to 0)
        '''
        eps = 1e-20
        y = tf.math.log(logits + eps) + self.sample_gumbel(tf.shape(logits))
        y = tf.nn.softmax(y / temperature)
        if self.params['hard']:
            y_hard = tf.cast(tf.equal(y, tf.math.reduce_max(y, 1, keepdims=True)), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        return y
    
    def decoder(self, x):
        h = self.dec_dense1(x)
        h = tf.nn.leaky_relu(h, alpha=0.1)
        h = self.dec_dense2(h)
        return h

    def call(self, x, tau):
        class_num = self.params["class_num"]   
        latent_dim = self.params["latent_dim"]   

        # encoder
        x = self.enc_dense1(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        
        # continous latent
        mean = layers.Concatenate(axis=1)([d(x)[:, tf.newaxis, :] for d in self.mean_layer])
        logvar = layers.Concatenate(axis=1)([d(x)[:, tf.newaxis, :] for d in self.logvar_layer])
        # epsilon = tf.random.normal((self.params["batch_size"], class_num, latent_dim))
        epsilon = tf.random.normal((x.shape[0], class_num, latent_dim))
        z = mean + tf.math.exp(logvar / 2) * epsilon 
        # assert z.shape == (self.params["batch_size"], class_num, latent_dim)
        assert z.shape == (x.shape[0], class_num, latent_dim)
        
        # discrete latent
        logits = self.logits(x)
        y = self.gumbel_softmax_sample(logits, tau)
        # assert y.shape == (self.params["batch_size"], class_num)
        assert y.shape == (x.shape[0], class_num)
        # self.sample_y = y
        
        # mixture sampling
        z_tilde = tf.squeeze(tf.matmul(y[:, tf.newaxis, :], z), axis=1)
        # assert z_tilde.shape == (self.params["batch_size"], latent_dim)
        assert z_tilde.shape == (x.shape[0], latent_dim)
                
        # decoder
        xhat = self.decoder(z_tilde) 
        # assert xhat.shape == (self.params["batch_size"], self.params['data_dim'])
        assert xhat.shape == (x.shape[0], self.params['data_dim'])
        
        return mean, logvar, logits, y, z, z_tilde, xhat
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
# also used for Kingma
def loss_unimodal(xhat, x, mean, logvar, beta, PARAMS):
    # reconstruction
    error = K.losses.mean_squared_error(x, xhat)
    error = 1/2 * tf.reduce_mean(error, 0, keepdims=True)    
    
    # KL loss by closed form
    kl = tf.reduce_mean(
        tf.reduce_sum(0.5 * (1/PARAMS['sigma']*tf.math.pow(mean, 2) - 1 - tf.math.log(1/PARAMS['sigma']) + 1/PARAMS['sigma']*tf.math.exp(logvar) - logvar), axis=-1)
        )
        
    return error + beta * kl, error, kl
#%%
def loss_mixture(logits, xhat, x, mean, logvar, temperature, beta, PARAMS):
    # reconstruction
    error = K.losses.mean_squared_error(x, xhat)
    error = 1/2 * tf.reduce_mean(error, 0, keepdims=True)    
    
    # KL loss by closed form
    logits_ = logits / tf.tile(tf.reduce_sum(logits, axis=-1, keepdims=True), [1, PARAMS['class_num']])
    kl1 = tf.reduce_mean(tf.reduce_sum(logits_ * (tf.math.log(logits_ + 1e-20) - tf.math.log(1/PARAMS['class_num'])), axis=1), keepdims=True)
    kl2 = tf.reduce_mean(tf.reduce_sum(tf.multiply(
                        tf.reduce_sum(0.5 * (1/PARAMS['sigma']*tf.math.pow(mean - PARAMS['prior_means'], 2) - 1 - tf.math.log(1/PARAMS['sigma']) + 1/PARAMS['sigma']*tf.math.exp(logvar) - logvar), axis=-1), 
                        logits_), axis=-1))
    kl_loss = kl1 + kl2
    
    return error + beta * kl_loss, error, kl_loss
#%%
def loss_mixture2(logits, xhat, x, mean, logvar, temperature, beta, PARAMS):
    # reconstruction
    error = K.losses.mean_squared_error(x, xhat)
    error = 1/2 * tf.reduce_mean(error, 0, keepdims=True)    
    
    # KL loss by closed form
    logits_ = logits / tf.tile(tf.reduce_sum(logits, axis=-1, keepdims=True), [1, PARAMS['class_num']])
    kl1 = tf.reduce_mean(tf.reduce_sum(logits_ * (tf.math.log(logits_ + 1e-20) - tf.math.log(1/PARAMS['class_num'])), axis=1), keepdims=True)
    kl2 = tf.reduce_mean(tf.reduce_sum(tf.multiply(
                        tf.reduce_sum(0.5 * (1/PARAMS['sigma']*tf.math.pow(mean - PARAMS['prior_means'], 2) - 1 - tf.math.log(1/PARAMS['prior_vars']) + 1/PARAMS['prior_vars']*tf.math.exp(logvar) - logvar), axis=-1), 
                        logits_), axis=-1))
    kl_loss = kl1 + kl2
    
    return error + beta * kl_loss, error, kl_loss
#%%
def center_reconstruction(model, epoch, PARAMS):
    prior_sample = tf.cast(PARAMS['prior_means'].numpy()[0], tf.float32)
    center = model.decoder(prior_sample[tf.newaxis, ...]).numpy()
    plt.figure(figsize=(20, 10))
    for i in range(PARAMS['class_num']):
        plt.subplot(1, PARAMS['class_num'], i+1)
        plt.imshow(center[0, i, :].reshape(28, 28), cmap='gray')
        plt.title(i, fontsize=25)
        plt.axis('off')
        plt.tight_layout() 
    # if PARAMS['FashionMNIST']:
    #     plt.savefig('./result/mixturevae_fashionmnist_center_epoch{}.png'.format(epoch),
    #                 bbox_inches="tight", pad_inches=0.1)
    # else:
    #     plt.savefig('./result/mixturevae_mnist_center_epoch{}.png'.format(epoch),
    #                 bbox_inches="tight", pad_inches=0.1)
#%%
def example_reconstruction(train_x, y, xhat):
    y = y.numpy()               
    xhat = xhat.numpy()  
    
    plt.figure(figsize=(5, 10))
    for i in range(0, 15, 3):
        # input img
        plt.subplot(5, 3, i+1)
        plt.imshow(train_x[i+5, :].numpy().reshape(28, 28), cmap='gray')
        plt.axis('off')
        # code
        plt.subplot(5, 3, i+2)
        plt.imshow(y[[i+5], :]) 
        plt.axis('off')
        # output img
        plt.subplot(5, 3, i+3)
        plt.imshow(xhat[i+5, :,].reshape((28, 28)), cmap='gray')
        plt.axis('off')
    # if PARAMS['FashionMNIST']:
    #     plt.savefig('./result/mixturevae_fashionmnist_rebuilt_epoch{}.png'.format(epoch),
    #                 bbox_inches="tight", pad_inches=0.1)
    # else:
    #     plt.savefig('./result/mixturevae_mnist_rebuilt_epoch{}.png'.format(epoch),
    #                 bbox_inches="tight", pad_inches=0.1)
#%%
def z_space(z, PARAMS):
    zmat = z.numpy().reshape(PARAMS['batch_size']*PARAMS['class_num'], PARAMS['latent_dim'])
    zlabel = sum([list(range(PARAMS['class_num']))]*PARAMS['batch_size'], [])
    plt.figure(figsize=(15, 15))
    plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
    plt.scatter(zmat[:, 0], zmat[:, 1], c=zlabel, s=20, cmap=plt.cm.Reds, alpha=1)
    # plt.savefig('./result/zsample_{}.png'.format(key), 
    #             dpi=600, bbox_inches="tight", pad_inches=0)
#%%