import tensorflow as tf

import keras
print(tf.__version__)
print(keras.__version__)

# Print available devices
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# If you have a GPU, prints the gpu device name
if tf.test.gpu_device_name():
    print('\nDefault GPU Device: {}'.format(tf.test.gpu_device_name()))
