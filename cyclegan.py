# This code is heavily inspired by the works of
# https://github.com/eriklindernoren/Keras-GAN/blob/master/cyclegan
# under the MIT License Copyright (c) 2017 Erik Linder-Norén


from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
# This code is heavily inspired by the works of
# https://github.com/eriklindernoren/Keras-GAN/blob/master/cyclegan
# under the MIT License Copyright (c) 2017 Erik Linder-Norén

from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os

class CycleGAN():
    def __init__(self, g_AB=None, g_BA=None):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.data_loader = DataLoader(img_res=(self.img_rows, self.img_cols))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        if g_AB is not None:
            print("Resume training from ", g_AB)
            self.g_AB = load_model(g_AB, custom_objects={'InstanceNormalization': InstanceNormalization})
        else:
            self.g_AB = self.build_generator()
        if g_BA is not None:
            print("Resume training from ", g_BA)
            self.g_BA = load_model(g_BA, custom_objects={'InstanceNormalization': InstanceNormalization})
        else:
            self.g_BA = self.build_generator()

        # Input images from both domains
        shoe = Input(shape=self.img_shape)
        handbag = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_handbag = self.g_AB(shoe)
        fake_shoe = self.g_BA(handbag)
        # Translate images back to original domain
        reconstr_shoe = self.g_BA(fake_handbag)
        reconstr_handbag = self.g_AB(fake_shoe)
        # Identity mapping of images
        shoe_id = self.g_BA(shoe)
        handbag_id = self.g_AB(handbag)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_shoe)
        valid_B = self.d_B(fake_handbag)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[shoe, handbag],
                              outputs=[ valid_A, valid_B,
                                        reconstr_shoe, reconstr_handbag,
                                        shoe_id, handbag_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)

        # self.combined.summary()

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        # Upsampling
        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)
        model = Model(d0, output_img)
        
        # model.summary()
        return model

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        model = Model(img, validity)
        # model.summary()

        return model

    def train(self, epochs, batch_size=1, img_sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (shoes, handbags) in enumerate(self.data_loader.load_batch(batch_size)):
                # ------------- ---------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_handbag = self.g_AB.predict(shoes)
                fake_shoe = self.g_BA.predict(handbags)

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(shoes, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_shoe, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(handbags, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_handbag, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)


                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([shoes, handbags],
                                                        [valid, valid,
                                                        shoes, handbags,
                                                        shoes, handbags])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6])))

                # If at save interval => save generated image samples
                if batch_i % img_sample_interval == 0:
                    self.sample_images(start_time, epoch, batch_i)

            # Saving generator every epoch

            directory = "models"
            if not os.path.exists(directory):
                os.makedirs(directory)
            modelname_AB = 'models/cyclegan_gAB_ep{}.h5'.format(epoch)
            modelname_BA = 'models/cyclegan_gBA_ep{}.h5'.format(epoch)
            print("Saving generator models to disk as {} and {}".format(modelname_AB, modelname_BA))
            self.g_AB.save(modelname_AB)
            self.g_BA.save(modelname_BA)

    def sample_images(self, start_time, epoch, batch_i):

        directory = "figures/{}".format(start_time)
        if not os.path.exists(directory):
            os.makedirs(directory)


        shoes, handbags = self.data_loader.load_data(nb_samples=1)

        # Demo (for GIF)
        #shoes = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
        #handbags = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

        # Translate images to the other domain
        fake_handbag = self.g_AB.predict(shoes)
        fake_shoe = self.g_BA.predict(handbags)
        # Translate back to original domain
        reconstr_shoe = self.g_BA.predict(fake_handbag)
        reconstr_handbag = self.g_AB.predict(fake_shoe)

        gen_imgs = np.concatenate([shoes, fake_handbag, reconstr_shoe, handbags, fake_shoe, reconstr_handbag])

        # Rescale images [-1, 1] to [0, 255]
        gen_imgs = gen_imgs * 255/2 + 255/2  # Rescale pixel values
        gen_imgs = gen_imgs.astype(int)

        titles = ['Original', 'Translated', 'Reconstructed']
        r, c = 2, 3
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("figures/{}/cyclegan{}_{}".format(start_time, epoch, batch_i))
        plt.close()


if __name__ == '__main__':

    resume_training = True
    modelname_AB = 'models/cyclegan_gAB_ep4.h5'
    modelname_BA = 'models/cyclegan_gBA_ep4.h5'

    if resume_training:
        gan = CycleGAN(g_AB=modelname_AB, g_BA=modelname_BA)
        gan.train(epochs=200, batch_size=1, img_sample_interval=500)
    else:
        gan = CycleGAN()
        gan.train(epochs=200, batch_size=1, img_sample_interval=500)
