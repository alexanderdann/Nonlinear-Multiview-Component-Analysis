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
z_dim = 1
c_dim = 2
num_views = 2
epochs = 10

autoencoder_dims = [(1, None), (256, 'relu') , (1, 'relu')]

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


def train_neutral_network(epochs, num_views, num_channels, encoder_dims, decoder_dims):
    batched_X = BatchPreparation(batch_size=batch_size, samples=samples, data=X)
    batched_Y = BatchPreparation(batch_size=batch_size, samples=samples, data=Y)
    assert tf.shape(batched_X)[2] == tf.shape(batched_Y)[2]
    assert batch_size == tf.shape(batched_X)[1]

    batch_dims = tf.shape(batched_X)[0]
    data_dim = tf.shape(batched_X)[2]

    print('--- Information about the intermediate data ---\n')
    print(f'Amount of Batches: {batch_dims}\nBatch Size: {batch_size}\nData Dimension: {data_dim}\n')

    NCA_Class = NonlinearComponentAnalysis(num_views=num_views,
                                     num_channels=num_channels,
                                     encoder_layers=encoder_dims,
                                     decoder_layers=decoder_dims,
                                     batch_size=batch_size)
    NCA_Model = NCA_Class.NCA


    for batch_idx in range(batch_dims):
        chunkX = batched_X[batch_idx]
        chunkY = batched_Y[batch_idx]


        chunkXandY = tf.concat([chunkX, chunkY], 1)
        print(f'Dimensions of the Input Data: {tf.shape(chunkXandY[None])}\n')
        sliced_data = [tf.Variable(chunkXandY[:, i])[None] for i in range(2*data_dim)]
        print(tf.shape(sliced_data))

        #NCA.fit(sliced_data, batch_size=None, epochs=50)
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                tape.watch(sliced_data)
                output_of_encoders, output_of_decoders = NCA_Model(sliced_data)
                print(tf.shape(output_of_encoders))
                print(tf.shape(output_of_decoders[0][0]))
                loss1 = NCA_Class.loss(output_of_encoders[0][0], output_of_encoders[1][0],
                                      output_of_decoders[0][0], output_of_decoders[1][0],
                                      chunkX, chunkY)
                loss = NCA_Class.mse(output_of_encoders[0][0], output_of_encoders[1][0])
                print(f'WOHFGI {tf.shape(loss)}')
                print(f'Lossy {tf.shape(loss1)}')


            gradients = tape.gradient(loss1, NCA_Model.trainable_variables)
            #print(f'--{[var.name for var in tape.watched_variables()]}--')
            #print(f'Gradients: {gradients}')
            NCA_Class.optimizer.apply_gradients(zip(gradients, NCA_Model.trainable_variables))

        break



train_neutral_network(
    epochs=epochs,
    num_views=num_views,
    num_channels=z_dim+c_dim,
    encoder_dims=autoencoder_dims,
    decoder_dims=autoencoder_dims)








if __name__ == '__main__':
    print(tf.version)
