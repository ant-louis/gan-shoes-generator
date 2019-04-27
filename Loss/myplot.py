import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth



def plotLosses(data, g_title, d_title):
    plt.figure()
    plt.plot(range(data.shape[0]), data['d_loss'])
    plt.plot(range(data.shape[0]), smooth(data['d_loss'], 200), 'r-')

    plt.ylim([-50, 50])
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Discriminator loss', fontsize=12)

    plt.savefig(d_title)
    plt.show()

    plt.figure()
    plt.plot(range(data.shape[0]), data[' g_loss'])
    plt.plot(range(data.shape[0]), smooth(data[' g_loss'], 200), 'r-')

    plt.ylim([-50, 50])
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Generator loss', fontsize=12)

    plt.savefig(g_title)
    plt.show()




plotLosses(pd.read_csv('losscWgan.csv'), "CWGAN_gloss.png", "CWGAN_dloss.png")
plotLosses(pd.read_csv('lossCWGAN(30000).csv'), "CWGAN30000_gloss.png", "CWGAN30000_dloss.png")
plotLosses(pd.read_csv('lossWGAN.csv'), "WGAN_gloss.png", "WGAN_dloss.png")