# Large amount of credit goes to:
# https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py and
# https://github.com/eriklindernoren/Keras-GAN/blob/master/cgan/cgan.py
# which I've used as a reference for this implementation
# Author: Hanling Wang
# Date: 2018-11-21

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial

import tensorflow as tf
import keras.backend as K
import glob
import cv2
import time
import os
import matplotlib.pyplot as plt

import math

import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0"

config = tf.ConfigProto(device_count = {'GPU': 1 , 'CPU': 1}) 
sess = tf.Session(config=config) 
K.set_session(sess)



class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        global batch_size
        alpha = K.random_uniform((batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])
        
class CWGANGP():
    def __init__(self, epochs=100, batch_size=32, sample_interval=50):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.nclasses = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.losslog = []
        self.epochs = epochs
        self.batch_size = batch_size
        self.sample_interval = sample_interval
        
        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(100,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)
        
        
    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        generator = Sequential()
        depth = 816
        dim = 2
        dropout_rate = 0.5

        # In: 100 noise variables
        # Out: dim x dim x depth
        generator.add(Dense(dim*dim*depth, input_dim=100))
        generator.add(Reshape((dim, dim, depth)))
        generator.add(BatchNormalization(momentum=0.9))
        generator.add(LeakyReLU(alpha=0.2))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        generator.add(UpSampling2D())
        generator.add(Conv2DTranspose(filters=int(depth/2), kernel_size=5, strides=2, padding='same'))
        generator.add(BatchNormalization(momentum=0.9))
        generator.add(Dropout(rate=dropout_rate))
        generator.add(LeakyReLU(alpha=0.2))

        # In: 2*dim x 2*dim x depth/2
        # Out: 4*dim x 4*dim x depth/4
        generator.add(UpSampling2D())
        generator.add(Conv2DTranspose(filters=int(depth/4), kernel_size=5, strides=2, padding='same'))
        generator.add(BatchNormalization(momentum=0.9))
        generator.add(Dropout(rate=dropout_rate))
        generator.add(LeakyReLU(alpha=0.2))


        # In: 4*dim x 4*dim x depth/4
        # Out: 8*dim x 8*dim x depth/8
        generator.add(UpSampling2D())
        generator.add(Conv2DTranspose(filters=int(depth/8), kernel_size=5, strides=2, padding='same'))
        generator.add(BatchNormalization(momentum=0.9))
        generator.add(Dropout(rate=dropout_rate))
        generator.add(LeakyReLU(alpha=0.2))

        # Out: 128 x 128 x 3 color image
        generator.add(Conv2DTranspose(filters=3, kernel_size=5, padding='same'))
        generator.add(Activation('tanh'))

        print("GENERATOR NETWORK SHAPE")
        generator.summary()
        return generator

    def build_critic(self):

        discr = Sequential()
        depth = 32

        # In: 102 x 135 x 3, depth = 1
        discr.add(Conv2D(filters=depth*1, kernel_size=5, strides=2,data_format='channels_last', input_shape=self.img_shape, padding='same'))
        discr.add(LeakyReLU(alpha=0.2))

        discr.add(Conv2D(filters=depth*2, kernel_size=5, strides=2, padding='same'))
        #discr.add(BatchNormalization(momentum=0.9))
        discr.add(LeakyReLU(alpha=0.2))

        discr.add(Conv2D(filters=depth*4, kernel_size=5, strides=2, padding='same'))
        #discr.add(BatchNormalization(momentum=0.9))
        discr.add(LeakyReLU(alpha=0.2))

        discr.add(Conv2D(filters=depth*8, kernel_size=5, strides=2, padding='same'))
        #discr.add(BatchNormalization(momentum=0.9))
        discr.add(LeakyReLU(alpha=0.2))
        # Out: 1-dim probability
        discr.add(Flatten())
        discr.add(Dense(1))

        print("DISCRIMINATOR NETWORK SHAPE")
        discr.summary()

        return discr
      
      
    def createTS(self, nb_samples):
        print("Loading images ... \n")

        images = np.zeros((nb_samples, self.img_rows, self.img_cols, self.channels), dtype=np.float32)
        input_directory = '/content/gdrive/My Drive/Shoes_Generator/all_athletic'
        print("Pre-processing images...")
        i = 0
        for img in glob.glob("{}/*.jpg".format(input_directory)):
            try:
                shoe = cv2.imread(img)
                shoe = cv2.resize(shoe, (128, 128))
                
                #Normalize image between -1 and 1
                channel_0 = (shoe[:,:,0].astype('float32') - 255/2)/(255/2)
                channel_1 = (shoe[:,:,1].astype('float32') - 255/2)/(255/2)
                channel_2 = (shoe[:,:,2].astype('float32') - 255/2)/(255/2)
                norm_shoe = np.stack([channel_0, channel_1, channel_2], axis=-1)
                images[i,:,:,:]= norm_shoe
            
                i += 1
                if i%500 == 0:
                    print('Loaded {} images out of {}'.format(i, nb_samples))
            except:
                print("Passed: ",i)
                pass
                
            if i == nb_samples:
                break

        #print("Image size: ",images[10,:,:,:].shape)
        #print("Image example: ", images[10,30:40,30:40,0])
        #print("Rescaled image", images[10,30:40,30:40,0] * 255/2 + 255/2)

        return images
    

    def train(self):

        
        X_train = self.createTS(10000)
        t = time.time()
        
        # Adversarial ground truths
        valid = -np.ones((self.batch_size, 1))
        fake =  np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1)) # Dummy gt for gradient penalty
        for epoch in range(self.epochs):
            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], self.batch_size)
                imgs= X_train[idx,:,:,:]
                # labels = np.zeros((self.batch_size, 1))
                # Sample generator input
                noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
                with tf.device('/device:GPU:0'):
                  # Train the critic
                  d_loss = self.critic_model.train_on_batch([imgs, noise], [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------
            # sampled_labels = np.zeros((self.batch_size, 1))
            with tf.device('/device:GPU:0'):
              g_loss = self.generator_model.train_on_batch([noise], valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
            self.losslog.append([d_loss[0], g_loss])
            
            # If at save interval => save generated image samples
            if epoch % self.sample_interval == 0:
                self.plot_images(epoch, t)
                self.generator.save_weights('generator', overwrite=True)
                self.critic.save_weights('discriminator', overwrite=True)
                with open('/content/gdrive/My Drive/Shoes_Generator/cWGAN/loss.log', 'w') as f:
                    f.writelines('d_loss, g_loss\n')
                    for each in self.losslog:
                        f.writelines('%s, %s\n'%(each[0], each[1]))

    def plot_images(self, epoch, time):
       
        samples = 16
        image_dir = "/content/gdrive/My Drive/Shoes_Generator/cWGAN/images/shoes{}".format(time)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        filename = image_dir + "/shoes_{}.png".format(epoch)
        # Generate noise and create new fake image
        noise = np.random.standard_normal(size=[samples, 100])
        # sampled_labels = np.zeros((samples, 1))
        images = self.generator.predict([noise])

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(math.sqrt(samples), math.sqrt(samples), i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols, 3])
            image = image * 255/2 + 255/2  # Rescale pixel values
            plt.imshow(image.astype(np.uint8))
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')
        

if __name__ == '__main__':
    epochs = 30000
    batch_size = 32
    sample_interval = 50
    wgan = CWGANGP(epochs, batch_size, sample_interval)
    wgan.train()