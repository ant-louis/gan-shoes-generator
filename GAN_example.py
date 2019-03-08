# -*- coding: utf-8 -*-
# generate new kinds of pokemons

import os
import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Conv2DTranspose, \
                        Dropout, LeakyReLU, BatchNormalization, UpSampling2D, \
                        Flatten, Reshape

"""======================= DISCRIMINATOR ===================="""

D = Sequential()
depth = 64
dropout = 0.2

# In: 28 x 28 x 1, depth = 1
img_rows = 28
img_cols = 28
channel = 1
# Out: 14 x 14 x 1, depth=64
input_shape = (img_rows, img_cols, channel)
D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same'))
D.add(LeakyReLU(alpha=0.2))

D.add(Dropout(dropout))
D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
D.add(LeakyReLU(alpha=0.2))
D.add(Activation('sigmoid'))

D.add(Dropout(dropout))
D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
D.add(LeakyReLU(alpha=0.2))
D.add(Dropout(dropout))
D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
D.add(LeakyReLU(alpha=0.2))
D.add(Dropout(dropout))
# Out: 1-dim probability
D.add(Flatten())
D.add(Dense(1))
D.add(Activation('sigmoid'))
D.summary()

"""======================= GENERATOR ===================="""
G = Sequential()
dropout = 0.2
depth = 64+64+64+64
dim = 7
# In: 100
# Out: dim x dim x depth
G.add(Dense(dim*dim*depth, input_dim=100))
G.add(BatchNormalization(momentum=0.9))
G.add(Activation('relu'))
G.add(Reshape((dim, dim, depth)))
G.add(Dropout(dropout))

# In: dim x dim x depth
# Out: 2*dim x 2*dim x depth/2
G.add(UpSampling2D())
G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
G.add(BatchNormalization(momentum=0.9))
G.add(Activation('relu'))
G.add(UpSampling2D())
G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
G.add(BatchNormalization(momentum=0.9))
G.add(Activation('relu'))
G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
G.add(BatchNormalization(momentum=0.9))
G.add(Activation('relu'))

# Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
G.add(Conv2DTranspose(1, 5, padding='same'))
G.add(Activation('sigmoid'))
G.summary()