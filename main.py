import tensorflow as tf
import numpy as np
from models import BatchPreparation, NonlinearComponentAnalysis, CCA
import matplotlib.pyplot as plt
import time
import os
from TwoChannelModel import *

keys = time.asctime(time.localtime(time.time())).split()

# Please change your folder
path = '/Users/alexander/Documents/Uni/Work/NMCA/Simulation/' + str('-'.join(keys[0:3]))


try:
    os.makedirs(path)
    print(f'Path: {path} exists: {os.path.exists(path)}\n\n')
except:
    pass

rhos = [0.9, 0.75, 0.0]
batch_size = 128
samples = 1024
z_dim = 2
c_dim = 3
num_views = 2
epochs = 1000

assert z_dim == 2

autoencoder_dims = [(1, None), (256, 'relu'), (1, 'relu')]

# Choose Parabola or Gaussian for relationship between the latent sources
# If transformation = True => Y = g(As) where g is a non-linear function
TCM = TwoChannelModel(
    path=path,
    observations=samples,
    mixing_dim=int(z_dim + c_dim),
    shared_dim=z_dim,
    private_dim=c_dim,
    mode='Parabola',
    transformation=True,
    rhos=rhos)

X, Y, S_x, S_y, created_rhos = TCM.getitems()

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

    loss_arr = []
    for batch_idx in range(batch_dims):
        chunkX = batched_X[batch_idx]
        chunkY = batched_Y[batch_idx]

        chunkXandY = tf.concat([chunkX, chunkY], 1)
        print(f'\n\nDimensions of the Input Data: {tf.shape(chunkXandY[None])}\n')
        sliced_data = [chunkXandY[:, i][None] for i in range(2*data_dim)]
        print(tf.shape(sliced_data))

        for epoch in range(epochs):
            print(f'######## Batch {batch_idx+1}/{batch_dims} ########')
            print(f'######## Epoch {epoch+1}/{epochs} ########')
            with tf.GradientTape() as tape:
                tape.watch(sliced_data)
                output_of_encoders, output_of_decoders = NCA_Model(sliced_data)
                print(f'O1 {output_of_encoders[0]}')
                print(f'O2 {output_of_encoders[1]}')
                c_loss = NCA_Class.loss(output_of_encoders[0][0], output_of_encoders[1][0],
                                      output_of_decoders[0][0], output_of_decoders[1][0],
                                      chunkX, chunkY, batch_size)

            gradients = tape.gradient(c_loss, NCA_Model.trainable_variables)

            NCA_Class.optimizer.apply_gradients(zip(gradients, NCA_Model.trainable_variables))

    eval_data_np, test_sample = TCM.eval(batch_size, num_channels, data_dim)
    eval_data_tf = tf.convert_to_tensor(eval_data_np, dtype=tf.float32)
    output_of_encoders, output_of_decoders = NCA_Model([eval_data_tf[i] for i in range(2*data_dim)])

    fig, axes = plt.subplots(num_channels, num_views, figsize=(6, 9))

    for c in range(num_channels):
        for v in range(num_views):
            axes[c, v].title.set_text(f'$View {v} Channel {c}$')
            axes[c, v].plot(test_sample, output_of_encoders[v][0][:,c].numpy(), label=r'$\mathrm{f}\circledast\mathrm{g}$')
            if v == 0:
                axes[c, v].plot(test_sample, np.squeeze(eval_data_np[:5][c]), label=r'$\mathrm{g}$')
            elif v == 1:
                axes[c, v].plot(test_sample, np.squeeze(eval_data_np[5:][c]), label=r'$\mathrm{g}$')

            axes[c, v].legend()

    plt.tight_layout()
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
