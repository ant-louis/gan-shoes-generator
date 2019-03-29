
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";  

# Just disables the warning for CPU instruction set,
#  doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import random
import time
import cv2
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop

from tensorflow import ConfigProto, Session
from keras.backend import set_session

config = ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 
sess = Session(config=config) 
set_session(sess)

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class DCGAN:

    def __init__(self):
        self.IMROWS = 136
        self.IMCOLS = 136
        self.IMCHANNELS = 3
        # self.train = self.createTS()
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model


    def discriminator(self):
        if self.D:
            return self.D

        self.D = Sequential()
        depth = 32
        dropout = 0.4

        # In: 136 x 136 x 3, depth = 1
        # Out: 14 x 14 x 1, depth=64
        input_shape = (self.IMROWS, self.IMCOLS, self.IMCHANNELS)
        self.D.add(Conv2D(filters=depth*1, kernel_size=5, strides=2,data_format='channels_last', input_shape=input_shape, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(filters=depth*2, kernel_size=5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(filters=depth*4, kernel_size=5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Activation('sigmoid'))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(filters=depth*8, kernel_size=5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(filters=depth*16, kernel_size=5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))

        self.D.summary()

        return self.D


    def generator(self): 
        if self.G:
            return self.G

        self.G = Sequential()
        dropout = 0.4
        depth = 64+64+64+64
        dim = 17
        # In: 100 noise variables
        # Out: dim x dim x depth
        self.G.add(Dense(dim*dim*depth, input_dim=100))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(filters=int(depth/2), kernel_size=5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # In: 2*dim x 2*dim x depth/2
        # Out: 4*dim x 4*dim x depth/4
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(filters=int(depth/4), kernel_size=5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # In: 4*dim x 4*dim x depth/4
        # Out: 8*dim x 8*dim x depth/8
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(filters=int(depth/8), kernel_size=5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # In: 8*dim x 8*dim x depth/8
        # Out: 8*dim x 8*dim x depth/16
        self.G.add(Conv2DTranspose(filters=int(depth/16), kernel_size=5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # Out: 136 x 136 x 3 color image
        self.G.add(Conv2DTranspose(filters=3, kernel_size=5, padding='same'))
        self.G.add(Activation('sigmoid'))
        self.G.summary()

        return self.G


    def discriminator_model(self):

        if self.DM:
            return self.DM

        #  ============== DISCRIMINATOR MODEL ==========================
        # Since the output of the Discriminator is sigmoid, we use binary cross entropy for the loss. 
        # RMSProp as optimizer generates more realistic fake images compared to Adam for this case. 
        # Learning rate is 0.0008. Weight decay and clip value stabilize learning during the latter 
        # part of the training. You have to adjust the decay if you adjust the learning rate.

        optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return self.DM

    def adversarial_model(self):

        if self.AM:
            return self.AM
        #  ============== ADVERSARIAL MODEL ==========================
        # The training parameters are the same as in the Discriminator model except 
        # a reduced learning rate and corresponding weight decay.

        optimizer = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return self.AM


class SHOES_DCGAN(object):

    def __init__(self, nb_samples):
        self.img_rows = 136
        self.img_cols = 136
        self.channels = 3

        self.x_train = self.createTS(nb_samples)
        print("Training set size: ", self.x_train.shape)

        self.DCGAN = DCGAN()
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def createTS(self, nb_samples):
        print("Loading images ... \n")
        image_dir = 'ut-zap50k-images-square/all_images'
        image_names = ['ut-zap50k-images-square/all_images/{}'.format(i) for i in os.listdir(image_dir)]
        
        # Shorten training set for troubleshooting

        if nb_samples < len(image_names):
            image_names = image_names[:nb_samples]

        images = np.zeros((len(image_names), self.img_rows, self.img_cols, self.channels), dtype=np.float32)

        # Load images
        for n, image in enumerate(image_names):
            if n % 1000 == 0:
                print('Loaded {} images out of {}'.format(n, len(image_names)))
            images[n] = cv2.imread(image, cv2.IMREAD_COLOR)

        # print("Image size: ", np.asarray(images[0]).shape)
        return images

    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'shoes.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "shoes_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols, 3])
            plt.imshow(image)
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()


if __name__ == '__main__':
    nb_samples = 10000
    Shoes_dcgan = SHOES_DCGAN(nb_samples)
    timer = ElapsedTimer()
    Shoes_dcgan.train(train_steps=10000, batch_size=32, save_interval=500)
    timer.elapsed_time()
    Shoes_dcgan.plot_images(fake=True)
    Shoes_dcgan.plot_images(fake=False, samples=4, save2file=True)