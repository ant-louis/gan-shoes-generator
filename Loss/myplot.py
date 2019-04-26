import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

data = pd.read_csv('lossWGAN.csv')

print(data.shape)
plt.plot(range(29951), data[' g_loss'])
plt.plot(range(29951), smooth(data[' g_loss'], 200), 'r-')

plt.ylim([-50, 50])
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Discriminent Loss', fontsize=12)

plt.savefig("Dloss.png")
plt.show()
