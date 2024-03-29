
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Just disables the warning for CPU instruction set,
#  doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import random
import math
import time
import cv2
import glob
from matplotlib import pyplot as plt

import tensorflow as tf
import keras.backend as K

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard


config = tf.ConfigProto(device_count = {'GPU': 1 , 'CPU': 8}) 
sess = tf.Session(config=config) 
K.set_session(sess)

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

    def __init__(self, generator_model):
        self.IMROWS = 128
        self.IMCOLS = 128
        self.IMCHANNELS = 3

        self.discriminator = self.build_discriminator()
        if generator_model is None:
            self.generator = self.build_generator()
        else:
            print("Loading saved generator model")
            self.generator = generator_model
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model



    def build_discriminator(self):

        discr = Sequential()
        depth = 32

        # In: 102 x 135 x 3, depth = 1
        input_shape = (self.IMROWS, self.IMCOLS, self.IMCHANNELS)
        discr.add(Conv2D(filters=depth*1, kernel_size=5, strides=2,data_format='channels_last', input_shape=input_shape, padding='same'))
        discr.add(LeakyReLU(alpha=0.2))

        discr.add(Conv2D(filters=depth*2, kernel_size=5, strides=2, padding='same'))
        discr.add(BatchNormalization(momentum=0.9))
        discr.add(LeakyReLU(alpha=0.2))

        discr.add(Conv2D(filters=depth*4, kernel_size=5, strides=2, padding='same'))
        discr.add(BatchNormalization(momentum=0.9))
        discr.add(LeakyReLU(alpha=0.2))

        discr.add(Conv2D(filters=depth*8, kernel_size=5, strides=2, padding='same'))
        discr.add(BatchNormalization(momentum=0.9))
        discr.add(LeakyReLU(alpha=0.2))
        # Out: 1-dim probability
        discr.add(Flatten())
        discr.add(Dense(1))
        discr.add(Activation('sigmoid'))

        print("DISCRIMINATOR NETWORK SHAPE")
        discr.summary()

        return discr


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


    def discriminator_model(self):

        if self.DM:
            return self.DM

        optimizer = RMSprop(lr=0.00005, decay=3e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator)
        print("trainable discr weights before comp: ", len(self.discriminator.trainable_weights))
        self.DM.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])
        print("DISCRIMINATOR MODEL: ")
        self.DM.summary()

        print("trainable discr weights after comp: ", len(self.DM._collected_trainable_weights))
        return self.DM

    def adversarial_model(self):

        if self.AM:
            return self.AM

        optimizer = RMSprop(lr=0.00005, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator)
        print("trainable gen weights before comp: ", len(self.generator.trainable_weights))

        # Fix discriminator weights in adversarial model
        self.discriminator.trainable = False
        self.AM.add(self.discriminator)
        self.AM.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])
        print("ADVERSARIAL MODEL: ")
        self.AM.summary()

        print("trainable gen weights after comp: ", len(self.AM._collected_trainable_weights))

        return self.AM

    def wasserstein_loss(self, y_true, y_pred):
        return -K.mean(y_true * y_pred)


class SHOES_DCGAN(object):

    def __init__(self, nb_samples=None, generator_model=None):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3

        self.buildModel(generator_model)
        self.x_train = self.createTS(nb_samples)
        print("Training set size: ", self.x_train.shape)



    def buildModel(self, generator_model):
        gan = DCGAN(generator_model)
        self.discriminator =  gan.discriminator_model()
        self.adversarial = gan.adversarial_model()
        self.generator = gan.generator


    def createTS(self, nb_samples):
        print("Loading images ... \n")

        images = np.zeros((nb_samples, self.img_rows, self.img_cols, self.channels), dtype=np.float32)
        input_directory = 'all_athletic'
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

        # print("Image size: ",images[0,:,:,:].shape)
        # print("Image example: ", images[0,:10,:10,0])
        # print("Rescaled image", images[0,:10,:10,0] * 255/2 + 255/2)

        return images

    def train(self, train_steps=2000, batch_size=256, n_critic=5, save_interval=0, show_samples=16):
        

        # Transform train_on_batch return value
        # to dict expected by on_batch_end callback
        # for tensorboard
        def named_logs(model, logs):
            result = {}
            for name, value in zip(model.metrics_names, logs):
                result[name] = value
            return result
        
        curr_time = time.time()
        real_labels = - np.ones((batch_size,1)) # -1

        for i in range(train_steps):

            # # Tensorboard outputs graphs and other metrics
            # tensorboard_discr = TensorBoard(log_dir="logs_and_graphs/{}/logs/discriminator/step_{}".format(curr_time,i),  
            #                             histogram_freq=0,
            #                             batch_size=batch_size,
            #                             write_graph=True,
            #                             write_grads=True)
            # tensorboard_adver = TensorBoard(log_dir="logs_and_graphs/{}/logs/adversarial/step_{}".format(curr_time, i),  
            #                             histogram_freq=0,
            #                             batch_size=batch_size,
            #                             write_graph=True,
            #                             write_grads=True)
            # tensorboard_discr.set_model(self.discriminator)
            # tensorboard_adver.set_model(self.adversarial)

            discriminator_loss = 0
            discriminator_acc = 0

            # For each training iteration of the adversarial network, traing
            # the discriminator for `n_critic` iterations
            for _ in range(n_critic):
                #Sample real images from training, creating a minibatch
                real_images = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size), :, :, :]
                
                # Initialize noise and create fake image from generator
                noise = np.random.standard_normal(size=[batch_size, 100]) #Normal noise
                fake_images = self.generator.predict(noise)

                with tf.device('/device:GPU:0'):
                    # Train discriminator
                    real_loss, real_acc = self.discriminator.train_on_batch(real_images, real_labels)
                    fake_loss, fake_acc = self.discriminator.train_on_batch(fake_images, -real_labels)

                # Mean loss between fake and real
                discriminator_loss += 0.5 * (real_loss + fake_loss)
                discriminator_acc += 0.5 * (real_acc + fake_acc)

                # Clip discriminator weights to satisfy Lipschitz constraint
                clip_value = 0.01
                for layer in self.discriminator.layers:
                    weights = layer.get_weights()
                    weights = [np.clip(weight,
                                    -clip_value,
                                    clip_value) for weight in weights]
                    layer.set_weights(weights)

            # Take the average loss over the discriminator iterations
            discriminator_loss /= n_critic
            discriminator_acc /= n_critic

            # Train generator (as discriminator weights are fixed)
            noise = np.random.standard_normal(size=[batch_size, 100]) #Normal noise
            with tf.device('/device:GPU:0'):
                adversarial_loss, adversarial_acc = self.adversarial.train_on_batch(noise, real_labels)

            # Graphs discriminator metrics using tensorboard and log to console
            # tensorboard_discr.on_epoch_end(i, named_logs(self.discriminator, [discriminator_loss, discriminator_acc]))
            # tensorboard_adver.on_epoch_end(i, named_logs(self.adversarial, [adversarial_loss, adversarial_acc]))
            log_mesg = "%d: [D loss: %f, acc: %f]   [A loss: %f, acc: %f]" % (i, discriminator_loss, discriminator_acc, adversarial_loss, adversarial_acc)
            print(log_mesg)
            
            # Plot sample images during training
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(fake=True, save2file=True, samples=show_samples, step=i, time = curr_time)

            # Saving generator every 500 iterations
            if (i+1)%500==0:
                modelname = 'models/my_generator_{}.h5'.format(i+1)
                print("Saving generator model to disk as", modelname)
                self.generator.save(modelname)  # creates a HDF5 file 'my_model.h5'


    def plot_images(self, save2file=False, fake=True, samples=16, step=0, time=time.time()):
        directory = "figures/{}".format(time)
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = "figures/{}/shoes_true_{}.png".format(time,step)
        if fake:
            filename = "figures/{}/shoes_fake_{}.png".format(time,step)
            # Generate noise and create new fake image
            noise = np.random.standard_normal(size=[samples, 100])
            images = self.generator.predict(noise)
        else:
            # Take images from the training set
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(math.sqrt(samples), math.sqrt(samples), i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols, 3])
            image = image * 255/2 + 255/2  # Rescale pixel values
            plt.imshow(image.astype(np.uint8))
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.clf()
            plt.show()


if __name__ == '__main__':
    # Shorten training set for troubleshooting
    NB_SAMPLES = 10000
    TRAINING_STEPS = 30000
    BATCH_SIZE = 16
    N_CRITIC = 5
    SAVE_INTERVAL = 100
    SHOW_SAMPLES = 4 # Squares only, e.g. 4 9 16 25 ..

    # my_model = load_model('models/my_generator_10000.h5')

    Shoes_dcgan = SHOES_DCGAN(nb_samples=NB_SAMPLES)
    timer = ElapsedTimer()
    Shoes_dcgan.train(TRAINING_STEPS, BATCH_SIZE, N_CRITIC, SAVE_INTERVAL, SHOW_SAMPLES)
    timer.elapsed_time()

