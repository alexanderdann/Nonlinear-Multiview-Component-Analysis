import tensorflow as tf
import numpy as np
from tqdm import tqdm
from models import BatchPreparation, NonlinearComponentAnalysis
import hickle as hkl
import time
import os

from metrics import dist
from TwoChannelModel import TwoChannelModel

keys = time.asctime(time.localtime(time.time())).split()

# Please change your folder
path = os.getcwd() + '/Final/' + str('-'.join(keys[0:3]))


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
epochs = 7500
assert z_dim == 2

#hidden_dims = [256, 128]
#lambda_regs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

hidden_dims = [256]
lambda_regs = [0.005]

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

X, Y, S_x, S_y, created_rhos, _, _ = TCM.getitems()

EVAL_TCM = TwoChannelModel(
    path=path,
    observations=samples,
    mixing_dim=int(z_dim + c_dim),
    shared_dim=z_dim,
    private_dim=c_dim,
    mode='Parabola',
    transformation=True,
    rhos=rhos,
    seed=9999)

X_e, Y_e, S_x_e, S_y_e, created_rhos_e, AS_x_e, AS_y_e = EVAL_TCM.getitems()


def train_neutral_network(epochs, num_views, num_channels, encoder_dims, decoder_dims, samples, dir_path, shared_dim, sources, lambda_reg):
    batched_X = BatchPreparation(batch_size=batch_size, samples=samples, data=X)
    batched_Y = BatchPreparation(batch_size=batch_size, samples=samples, data=Y)

    shared_Sx, shared_Sy = tf.convert_to_tensor(sources[0][:shared_dim]), tf.convert_to_tensor(sources[1][:shared_dim])

    assert tf.shape(batched_X)[2] == tf.shape(batched_Y)[2]
    assert batch_size == tf.shape(batched_X)[2]

    batch_dims = tf.shape(batched_X)[0]
    data_dim = tf.shape(batched_X)[1]

    print('--- Information about the intermediate data ---\n')
    print(f'Amount of Batches: {batch_dims}\nBatch Size: {batch_size}\nData Dimension: {data_dim}\n')

    NCA_Class = NonlinearComponentAnalysis(num_views=num_views,
                                           num_channels=num_channels,
                                           encoder_layers=encoder_dims,
                                           decoder_layers=decoder_dims,
                                           batch_size=batch_size,
                                           lambda_val=lambda_reg)
    NCA_Model = NCA_Class.NCA

    loss_arr = []
    loss_arr_1 = []
    loss_arr_2 = []
    dist_arr = []

    for batch_idx in tqdm(range(batch_dims), desc='Batch ID'):
        chunkX = tf.transpose(batched_X[batch_idx])
        chunkY = tf.transpose(batched_Y[batch_idx])
        chunkXandY = tf.concat([chunkX, chunkY], 1)

        data_chunk = [chunkXandY[:, i][None] for i in range(2*data_dim)]

        for epoch in tqdm(range(epochs), desc='Epochs'):
            #print(f'######## Batch {batch_idx+1}/{batch_dims} ########')
            #print(f'######## Epoch {epoch+1}/{epochs} ########')
            with tf.GradientTape() as tape:
                tape.watch(data_chunk)
                output_of_encoders, output_of_decoders = NCA_Model(data_chunk)

                c_loss, loss1, loss2, U_dist = NCA_Class.loss(output_of_encoders[0][0], output_of_encoders[1][0],
                                      output_of_decoders[0][0], output_of_decoders[1][0],
                                      chunkX, chunkY, batch_size, shared_dim)
                loss_arr.append(c_loss)
                loss_arr_1.append(loss1)
                loss_arr_2.append(loss2)

            gradients = tape.gradient(c_loss, NCA_Model.trainable_variables)
            NCA_Class.optimizer.apply_gradients(zip(gradients, NCA_Model.trainable_variables))


            try:
                dist_val = dist(shared_Sx, U_dist, samples)
                dist_arr.append(dist_val)

            except:
                dist_val = -1
                dist_arr.append(dist_val)

    NCA_Model.save(f'{dir_path}/Model')

    hkl.dump([loss_arr, loss_arr_1, loss_arr_2, dist_arr,
             batch_size, num_channels, num_views, shared_dim, data_dim,
             epochs, samples, NCA_Class, X_e, Y_e, S_x_e, S_y_e, AS_x_e, AS_y_e], f'{dir_path}/params.p')

for it in range(0, 3):
    for hidd_dim in hidden_dims:
        for lamb_val in lambda_regs:
            autoencoder_dims = [(1, None), (hidd_dim, 'relu'), (1, None)]
            path_add = f'/Iteration {it}/Hidden Dimension {hidd_dim}/lambda {lamb_val}'
            full_path = path+path_add

            try:
                os.makedirs(full_path)
                print(f'Path: {path} exists: {os.path.exists(path)}\n\n')
            except:
                pass

            try:
                train_neutral_network(epochs, num_views, z_dim+c_dim, autoencoder_dims, autoencoder_dims,
                                  samples, full_path, z_dim, [S_x, S_y], lamb_val)

            except:
                print(f'ERROR in Iteration {it} Lambda {lamb_val} Hidden Dim {hidd_dim}')

if __name__ == '__main__':
    print(tf.version)
