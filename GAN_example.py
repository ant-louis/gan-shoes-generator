
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
import glob
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop

from tensorflow import ConfigProto, Session
from keras.backend import set_session
from keras.callbacks import TensorBoard

import keras.backend

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

def imagePreprocessing():

    input_directory = 'all_athletic'
    output_directory = 'training'
    print("Pre-processing images")
    i = 0
    for img in glob.glob("{}/*.jpg".format(input_directory)):
        try:
            shoe = cv2.imread(img)
            shoe = cv2.resize(shoe, (128, 128))

            #Normalize image
            mean, std = cv2.meanStdDev(shoe)
            channel_0 = shoe[:,:,0].astype('float32')/255
            channel_1 = shoe[:,:,1].astype('float32')/255
            channel_2 = shoe[:,:,2].astype('float32')/255
            norm_shoe = np.stack([channel_0, channel_1, channel_2], axis=-1)

            cv2.imwrite("{0}/img{1:0>5}.jpg".format(output_directory, i), norm_shoe)
            
            i += 1
            if i%500 == 0:
                print(i)
        except:
            print("Passed: ",i)
            pass


class DCGAN:

    def __init__(self):
        self.IMROWS = 128
        self.IMCOLS = 128
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

        # In: 102 x 135 x 3, depth = 1
        input_shape = (self.IMROWS, self.IMCOLS, self.IMCHANNELS)
        self.D.add(Conv2D(filters=depth*1, kernel_size=5, strides=2,data_format='channels_last', input_shape=input_shape, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))

        self.D.add(Conv2D(filters=depth*2, kernel_size=5, strides=2, padding='same'))
        self.D.add(BatchNormalization(momentum=0.9))
        self.D.add(LeakyReLU(alpha=0.2))

        self.D.add(Conv2D(filters=depth*4, kernel_size=5, strides=2, padding='same'))
        self.D.add(BatchNormalization(momentum=0.9))
        self.D.add(LeakyReLU(alpha=0.2))

        self.D.add(Conv2D(filters=depth*8, kernel_size=5, strides=2, padding='same'))
        self.D.add(BatchNormalization(momentum=0.9))
        self.D.add(LeakyReLU(alpha=0.2))

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
        depth = 816
        dim = 2
        dropout_rate = 0.5

        # In: 100 noise variables
        # Out: dim x dim x depth
        self.G.add(Dense(dim*dim*depth, input_dim=100))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(LeakyReLU(alpha=0.2))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(filters=int(depth/2), kernel_size=5, strides=2, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Dropout(rate=dropout_rate))
        self.G.add(LeakyReLU(alpha=0.2))

        # In: 2*dim x 2*dim x depth/2
        # Out: 4*dim x 4*dim x depth/4
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(filters=int(depth/4), kernel_size=5, strides=2, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Dropout(rate=dropout_rate))
        self.G.add(LeakyReLU(alpha=0.2))


        # In: 4*dim x 4*dim x depth/4
        # Out: 8*dim x 8*dim x depth/8
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(filters=int(depth/8), kernel_size=5, strides=2, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Dropout(rate=dropout_rate))
        self.G.add(LeakyReLU(alpha=0.2))

        # Out: 128 x 128 x 3 color image
        self.G.add(Conv2DTranspose(filters=3, kernel_size=5, padding='same'))
        self.G.add(Activation('tanh'))
        self.G.summary()

        return self.G


    def discriminator_model(self):

        if self.DM:
            return self.DM

        optimizer = RMSprop(lr=0.00005, clipvalue=0.01, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])

        return self.DM

    def adversarial_model(self):

        if self.AM:
            return self.AM

        optimizer = RMSprop(lr=0.00005, clipvalue=0.01, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])

        return self.AM

    def wasserstein_loss(self, y_true, y_pred):
        return keras.backend.mean(y_true * y_pred)


class SHOES_DCGAN(object):

    def __init__(self, nb_samples=None):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3

        self.x_train = self.createTS(nb_samples)
        print("Training set size: ", self.x_train.shape)

        self.DCGAN = DCGAN()
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def createTS(self, nb_samples):
        print("Loading images ... \n")
        image_dir = 'training'
        image_names = ['{}/{}'.format(image_dir, i) for i in os.listdir(image_dir)]
        
        if nb_samples is not None:
            if nb_samples < len(image_names):
                image_names = image_names[:nb_samples]

        images = np.zeros((len(image_names), self.img_rows, self.img_cols, self.channels), dtype=np.float32)

        # Load images
        for n, image in enumerate(image_names):
            if n % 1000 == 0:
                print('Loaded {} images out of {}'.format(n, len(image_names)))
                # plt.figure()
                # plt.imshow(cv2.imread(image, cv2.IMREAD_COLOR))
                # plt.show()
            images[n] = cv2.imread(image, cv2.IMREAD_COLOR)

        # print("Image size: ", np.asarray(images[0]).shape)
        return images

    def train(self, train_steps=2000, batch_size=256, save_interval=0, show_samples=16):
        curr_time = time.time()

        for i in range(train_steps):

            # Transform train_on_batch return value
            # to dict expected by on_batch_end callback
            def named_logs(model, logs):
                result = {}
                for name, value in zip(model.metrics_names, logs):
                    result[name] = value
                return result
            
            # Tensorboard outputs graphs and other metrics
            tensorboard_discr = TensorBoard(log_dir="logs_and_graphs/{}/discriminator/step_{}".format(curr_time,i),  
                                        histogram_freq=0,
                                        batch_size=batch_size,
                                        write_graph=True,
                                        write_grads=True)
            tensorboard_adver = TensorBoard(log_dir="logs_and_graphs/{}/adversarial/step_{}".format(curr_time, i),  
                                        histogram_freq=0,
                                        batch_size=batch_size,
                                        write_graph=True,
                                        write_grads=True)
            tensorboard_discr.set_model(self.discriminator)
            tensorboard_adver.set_model(self.adversarial)


            #Sample real images from training, creating a minibatch
            real_images = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size), :, :, :]

            real_labels = - np.ones((batch_size,1)) # -1
            
            # Initialize noise and create fake image from generator
            mu, sigma = 0, 1
            noise = np.random.normal(mu, sigma, size=[batch_size, 100]) #Normal noise
            fake_images = self.generator.predict(noise)


            # Train discriminator
            real_loss, real_acc = self.discriminator.train_on_batch(real_images, real_labels)
            fake_loss, fake_acc = self.discriminator.train_on_batch(fake_images, - real_labels)

            # Mean loss between fake and real
            discriminator_loss = 0.5 * (real_loss + fake_loss)
            discriminator_acc = 0.5 * (real_acc + fake_acc)

            # Train generator (as discriminator weights are fixed)
            adversarial_loss, adversarial_acc = self.adversarial.train_on_batch(noise, real_labels)

            # # When training adversarial alone :
            # print("{}: [A loss: {}, acc: {}]".format(i, adversarial_logs[0], adversarial_logs[1]))

            log_mesg = "%d: [D loss: %f, acc: %f]   [A loss: %f, acc: %f]" % (i, discriminator_loss, discriminator_acc, adversarial_loss, adversarial_acc)
            print(log_mesg)

            # Graphs discriminator metrics using tensorboard
            tensorboard_discr.on_epoch_end(i, named_logs(self.discriminator, [discriminator_loss, discriminator_acc]))
            tensorboard_adver.on_epoch_end(i, named_logs(self.adversarial, [adversarial_loss, adversarial_acc]))
            
            # Plot sample images during training
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(fake=True, save2file=True, samples=show_samples, step=i)

    def plot_images(self, save2file=False, fake=True, samples=16, step=0):
        filename = "shoes_true_{}.png".format(step)
        if fake:
            # Default noise
            mu, sigma = 0, 1
            noise = np.random.normal(mu, sigma, size=[samples, 100])
            filename = "shoes_fake_{}.png".format(step)

            # Do the prediction
            images = self.generator.predict(noise)
        else:
            # Generate `samples` number of samples
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols, 3])
            image = image * 255 # Rescale pixel values
            plt.imshow(image.astype(np.uint8))
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()


if __name__ == '__main__':
    # Shorten training set for troubleshooting
    nb_samples = 10000

    Shoes_dcgan = SHOES_DCGAN(nb_samples)
    timer = ElapsedTimer()
    Shoes_dcgan.train(train_steps=10000, batch_size=32, save_interval=1, show_samples=16)
    timer.elapsed_time()

    # # Preprocess images once, they are saved in directory 'training" 
    # imagePreprocessing()

