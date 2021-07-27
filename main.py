import tensorflow as tf
import numpy as np
from models import MiniMaxCCA, CNNdecoder, CNNencoder, CNNDAE, CCA
import time
import os
from TwoChannelModel import *

keys = time.asctime(time.localtime(time.time())).split()

path = '/Users/alexander/Documents/Uni/Work/deepCCA/Simulation/' + str('-'.join(keys))

try:
    os.makedirs(path)
    print(f'Path: {path} exists: {os.path.exists(path)}\n\n')
except:
    pass

rhos = [0.9, 0.15, 0.0]
part_mix = [0, 1, 2]

X, Y, S_x, S_y, created_rhos = TwoChannelModel(path, rhos, 1000, 500).transform(part_mix)

CCA(X, Y, 3)

if __name__ == '__main__':
    print(tf.version)
