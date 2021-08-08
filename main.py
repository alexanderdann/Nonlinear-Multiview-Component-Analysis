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


def train_neutral_network(num_views, encoder_dims, decoder_dims):
    batched_X = BatchPreparation(batch_size=batch_size, samples=samples, data=X)
    batched_Y = BatchPreparation(batch_size=batch_size, samples=samples, data=Y)
    assert tf.shape(batched_X)[2] == tf.shape(batched_Y)[2]

    batch_dims = tf.shape(batched_X)[0]
    batch_length = tf.shape(batched_X)[1]
    data_dim = tf.shape(batched_X)[2]

    for batch_idx in range(batch_dims):
        for ind in range(data_dim):
            chunkX = tf.reshape(
                tf.transpose(batched_X[batch_idx])[ind],
                [batch_length, 1]
            )

            chunkY = tf.reshape(
                tf.transpose(batched_Y[batch_idx])[ind],
                [batch_length, 1]
            )

            NN = NonlinearComponentAnalysis(num_views=num_views,
                                       encoder_layers=encoder_dims,
                                       decoder_layers=decoder_dims
                                       )

            with tf.GradientTape() as tape:
                print(chunkX)
                enc_output_1 = NN.model1_enc(chunkX, training=True)
                enc_output_2 = NN.model2_enc(chunkY, training=True)

                print(f' ENCODER : {enc_output_2}')

                dec_output_1 = NN.model1_dec(enc_output_1, training=True)
                dec_output_2 = NN.model2_dec(enc_output_2, training=True)

                print(f' DECODER OUTPUT : {enc_output_2}')

                loss_value = NN.loss(enc_output_1,
                                     enc_output_2,
                                     dec_output_1,
                                     dec_output_2,
                                     chunkX,
                                     chunkY)

            grads1 = tape.gradient(loss_value, NN.model1.trainable_variables)
            NN.optimizer1.apply_gradients(zip(grads1, NN.model1.trainable_variables))
            grads2 = tape.gradient(loss_value, NN.model2.trainable_variables)
            NN.optimizer2.apply_gradients(zip(grads2, NN.model2.trainable_variables))




train_neutral_network(num_views=num_views, encoder_dims=autoencoder_dims, decoder_dims=autoencoder_dims)








if __name__ == '__main__':
    print(tf.version)
