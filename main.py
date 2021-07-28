import tensorflow as tf
import numpy as np
from models import MiniMaxCCA, Decoder, Encoder, DeepAutoencoder, CCA
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

rhos = [0.9, 0.15, 0.0, 0.0, 0.0]
part_mix = [0, 1, 2]

# Choose Parabola or Gaussian for relationship between the latent sources
# If transformation = True => Y = g(As) where g is a non-linear function
X, Y, S_x, S_y, created_rhos = TwoChannelModel(path, 1000, 5, 2, 3, 'Parabola', True, rhos).getitems()

CCA(X, Y, 3)

def trainModel(epochs, batch_size, shared_dim, private_dim, stages):
    DAE =



if __name__ == '__main__':
    print(tf.version)
