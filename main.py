import tensorflow as tf
import numpy as np
from models import BatchPreparation, NonlinearComponentAnalysis
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
batch_size = 64
samples = 1024
z_dim = 2
c_dim = 3
num_views = 2

autoencoder_dims = [(1, None), (256, 'relu'), (256, 'relu'), (1, 'relu')]

# Choose Parabola or Gaussian for relationship between the latent sources
# If transformation = True => Y = g(As) where g is a non-linear function
X, Y, S_x, S_y, created_rhos = TwoChannelModel(
    path=path,
    observations=samples,
    mixing_dim=int(z_dim + c_dim),
    shared_dim=z_dim,
    private_dim=c_dim,
    mode='Parabola',
    transformation=True,
    rhos=rhos).getitems()


def train_neutral_network(num_views, num_channels, encoder_dims, decoder_dims):
    batched_X = BatchPreparation(batch_size=batch_size, samples=samples, data=X)
    batched_Y = BatchPreparation(batch_size=batch_size, samples=samples, data=Y)
    assert tf.shape(batched_X)[2] == tf.shape(batched_Y)[2]
    assert batch_size == tf.shape(batched_X)[1]

    batch_dims = tf.shape(batched_X)[0]
    data_dim = tf.shape(batched_X)[2]

    print('--- Information about the intermediate data ---\n')
    print(f'Amount of Batches: {batch_dims}\nBatch Size: {batch_size}\nData Dimension: {data_dim}\n')

    NCA = NonlinearComponentAnalysis(num_views=num_views,
                                     num_channels=num_channels,
                                     encoder_layers=encoder_dims,
                                     decoder_layers=decoder_dims,
                                     batch_size=batch_size).NCA


    for batch_idx in range(batch_dims):
        print(f'CATCH {tf.shape(batched_X)}')
        chunkX = batched_X[batch_idx]
        print(f'ss{tf.shape(chunkX)}')
        chunkY = batched_Y[batch_idx]


        chunkXandY = tf.concat([chunkX, chunkY], 1)
        print(f'Dimensions of the Input Data: {tf.shape(chunkXandY[None])}\n')
        sliced_data = [chunkXandY[:, i][None] for i in range(2*data_dim)]
        print(tf.shape(sliced_data))
        NCA.fit(sliced_data, batch_size=None)



train_neutral_network(
    num_views=num_views,
    num_channels=z_dim+c_dim,
    encoder_dims=autoencoder_dims,
    decoder_dims=autoencoder_dims)








if __name__ == '__main__':
    print(tf.version)
