import tensorflow as tf
import numpy as np
from models import BatchPreparation, NonlinearComponentAnalysis, CCA
import matplotlib.pyplot as plt
import time
import os
from TwoChannelModel import *

keys = time.asctime(time.localtime(time.time())).split()

path = '/Users/alexander/Documents/Uni/Work/NMCA/Simulation/' + str('-'.join(keys[0:3]))


try:
    os.makedirs(path)
    print(f'Path: {path} exists: {os.path.exists(path)}\n\n')
except:
    pass

rhos = [0.9, 0.75, 0.0]
batch_size = 1024
samples = 1024
z_dim = 2
c_dim = 3
num_views = 2
epochs = 10

assert z_dim == 2

autoencoder_dims = [(1, None), (256, 'relu'), (1, 'relu')]

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

#X1, Y1, S_x1, S_y1, created_rhos1 = TwoChannelModel(
#    path=path,
#    observations=samples,
#    mixing_dim=int(z_dim + c_dim),
#    shared_dim=z_dim,
#    private_dim=c_dim,
#    mode='Gaussian',
#    transformation=False,
#    rhos=rhos).getitems()
#
#result = CCA(X1, Y1).getitems()


def train_neutral_network(epochs, num_views, num_channels, encoder_dims, decoder_dims, samples, plot_path):
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
        print(f'\n\nDimensions of the Input Data: {tf.shape(chunkXandY[None])}\n')
        sliced_data = [chunkXandY[:, i][None] for i in range(2*data_dim)]
        print(tf.shape(sliced_data))

        #NCA.fit(sliced_data, batch_size=None, epochs=50)
        for epoch in range(epochs):
            print(f'######## Batch {batch_idx+1}/{batch_dims} ########')
            print(f'######## Epoch {epoch}/{epochs} ########')
            with tf.GradientTape() as tape:
                tape.watch(sliced_data)
                output_of_encoders, output_of_decoders = NCA_Model(sliced_data)
                print(tf.shape(output_of_encoders))
                print(tf.shape(output_of_decoders[0][0]))
                c_loss = NCA_Class.loss(output_of_encoders[0][0], output_of_encoders[1][0],
                                      output_of_decoders[0][0], output_of_decoders[1][0],
                                      chunkX, chunkY, batch_size)
                #tf_loss = NCA_Class.mse(output_of_encoders[0][0], output_of_encoders[1][0])
                #print(f'Custom: {tf.shape(c_loss)}')
                #print(f'tf.keras.losses: {tf.shape(tf_loss)}')


            gradients = tape.gradient(c_loss, NCA_Model.trainable_variables)
            #print(f'--{[var.name for var in tape.watched_variables()]}--')
            #print(f'Gradients: {gradients}')
            NCA_Class.optimizer.apply_gradients(zip(gradients, NCA_Model.trainable_variables))

    test_sample = tf.linspace(-10, 10, batch_size)
    tile_dim = tf.cast([data_dim, 1], dtype=tf.int32)
    data1 = tf.tile(test_sample[None], tile_dim)
    data2 = tf.tile(test_sample[None], tile_dim)
    dd = tf.concat([data1, data2], 0)
    aa = [dd[i][None] for i in range(2 * data_dim)]
    output_of_encoders, output_of_decoders = NCA_Model(aa)

    fig, axes = plt.subplots(num_channels, num_views, figsize=(6, 9))
    for c in range(num_channels):
        for v in range(num_views):
            axes[c, v].title.set_text(f'$View {v} Channel {c}$')
            Xd = np.copy(test_sample)
            Yd = np.copy(test_sample)
            _sigmoid = lambda x: np.array([1 / (1 + np.exp(-x_i)) for x_i in x])

            i=c
            if i == 0:
                Xd = 3 * _sigmoid(Xd) + 0.1 * Xd
                Yd = 5 * np.tanh(Yd) + 0.2 * Yd
            elif i == 1:
                Xd = 5 * _sigmoid(Xd) + 0.2 * Xd
                Yd = 2 * np.tanh(Yd) + 0.1 * Yd
            elif i == 2:
                Xd = 0.2 * np.exp(Xd)
                Yd = 0.1 * Yd ** 3 + Yd
            elif i == 3:
                Xd = -4 * _sigmoid(Xd) - 0.3 * Xd
                Yd = -5 * np.tanh(Yd) - 0.4 * Yd
            elif i == 4:
                Xd = -3 * _sigmoid(Xd) + 0.2 * Xd
                Yd = -6 * np.tanh(Yd) - 0.3 * Yd
            else:
                break

            if v == 0:
                axes[c, v].plot(test_sample, Xd, label=r'$\mathrm{g}$')
                res = [output_of_encoders[v][0][:, c].numpy()[i] * Xd[i] for i in range(len(test_sample))]
                axes[c, v].plot(test_sample, res, label=r'$\mathrm{f}\circledast\mathrm{g}$')
            elif v == 1:
                axes[c, v].plot(test_sample, Yd, label=r'$\mathrm{g}$')
                res = [output_of_encoders[v][0][:, c].numpy()[i] * Yd[i] for i in range(len(test_sample))]
                axes[c, v].plot(test_sample, res, label=r'$\mathrm{f}\circledast\mathrm{g}$')

            axes[c, v].plot(test_sample, output_of_encoders[v][0][:,c].numpy(), label=r'$\mathrm{f}$')
            axes[c, v].legend()
            plt.tight_layout()
            print([i==0 for i in output_of_encoders[v][0][:,c]])
            print(f'{v} {c}: {output_of_encoders[v][0][:,c]}')
            print(f'{v} {c}: {output_of_decoders[v][0][:, c]}')

    full_path = path + '/' + plot_path + f'_{epochs}_Epochs.png'
    plt.savefig(full_path)
    plt.show()



for it in range(1):
    path_add = f'Iteration_{it}'
    train_neutral_network(
        epochs=epochs,
        num_views=num_views,
        num_channels=z_dim+c_dim,
        encoder_dims=autoencoder_dims,
        decoder_dims=autoencoder_dims,
        samples=samples,
        plot_path=path_add)

if __name__ == '__main__':
    print(tf.version)
