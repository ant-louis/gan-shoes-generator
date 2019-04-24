# This code is heavily inspired by the works of
# https://github.com/eriklindernoren/Keras-GAN/blob/master/cyclegan
# under the MIT License Copyright (c) 2017 Erik Linder-Norén

import scipy
import cv2
from matplotlib import pyplot as plt
from glob import glob
import numpy as np

class DataLoader():
    def __init__(self, img_res=(128, 128)):
        self.img_res = img_res

    def load_data(self, nb_samples=1):
        
        handbag_path = glob('dataset/edges2handbags_sample/*.jpg')
        shoe_path = glob("dataset/shoes-athletic/*.jpg")

        handbags = np.zeros((nb_samples, self.img_res[0], self.img_res[1], 3), dtype=np.float32)
        shoes = np.zeros((nb_samples, self.img_res[0], self.img_res[1], 3), dtype=np.float32)

        for i in range(nb_samples):
            handbag = handbag_path[i + 200] # 200th shoe/handbag because 1st are same color
            shoe = shoe_path[i + 200]

            # Importing handbags
            try:
                handbag = self.imread(handbag)
                h, w, _ = handbag.shape

                # Separate edge and image
                half_w = int(w/2) 
                # handbag_edge = handbag[:, :half_w, :]
                handbag_img = handbag[:, half_w:, :]

                # handbag_edge = cv2.resize(handbag_edge, self.img_res)
                handbags[i,:,:,:] = cv2.resize(handbag_img, self.img_res)

            except Exception as e:
                print("Couldn't load handbag")
                print(e)

            # Importing shoes
            try:
                shoe_img = self.imread(shoe)
                shoes[i,:,:,:] = cv2.resize(shoe_img, self.img_res)
            except Exception as e:
                print("Couldn't load shoe")
                print(e)


        # Normalizing arrays to [-1, 1]
        shoes_norm = (shoes - 255/2)/(255/2)
        handbags_norm = (handbags - 255/2)/(255/2)

        return shoes_norm, handbags_norm

    def load_batch(self, batch_size=1):
        handbag_path = glob('./dataset/edges2handbags_sample/*.jpg')
        shoe_path = glob("./dataset/shoes-athletic/*.jpg")

        handbag_n_batches = int(len(handbag_path) / batch_size)
        shoe_n_batches = int(len(shoe_path) / batch_size)

        self.n_batches = handbag_n_batches if handbag_n_batches < shoe_n_batches else shoe_n_batches


        for i in range(self.n_batches-1):
            handbags = np.zeros((batch_size, self.img_res[0], self.img_res[1], 3), dtype=np.float32)
            shoes = np.zeros((batch_size, self.img_res[0], self.img_res[1], 3), dtype=np.float32)

            handbag_batch = handbag_path[i*batch_size:(i+1)*batch_size]
            shoe_batch = shoe_path[i*batch_size:(i+1)*batch_size]

            # Importing handbags
            for k,handbag in enumerate(handbag_batch):
                try:
                    handbag = self.imread(handbag)
                    h, w, _ = handbag.shape

                    # Separate edge and image
                    half_w = int(w/2) 
                    # handbag_edge = handbag[:, :half_w, :]
                    handbag_img = handbag[:, half_w:, :]

                    # handbag_edge = cv2.resize(handbag_edge, self.img_res)
                    handbags[k,:,:,:] = cv2.resize(handbag_img, self.img_res)

                except Exception as e:
                    print("Couldn't load handbag")
                    print(e)

            
            # Importing shoes
            for k, shoe in enumerate(shoe_batch):
                try:
                    shoe_img = self.imread(shoe)
                    shoes[k,:,:,:] = cv2.resize(shoe_img, self.img_res)
                except Exception as e:
                    print("Couldn't load shoe")
                    print(e)


            # Normalizing arrays to [-1, 1]
            shoes_norm = (shoes- 255/2)/(255/2)
            handbags_norm = (handbags - 255/2)/(255/2)

            yield shoes_norm, handbags_norm

    def imread(self, path):
        return cv2.imread(path)

if __name__=='__main__':
    dl = DataLoader()
    dl.load_data()
