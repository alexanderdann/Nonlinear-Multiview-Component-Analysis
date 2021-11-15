import os
import tensorflow as tf
from tqdm import tqdm
import nmca_model
from TwoChannelModel import TwoChannelModel
from correlation_analysis import CCA, PCC_Matrix
from tf_summary import write_image_summary, write_metric_summary, write_PCC_summary, write_gradients_summary


LOGPATH = f'{os.getcwd()}/LOG/POLY'
os.makedirs(LOGPATH)

# Generate data
#data_model = TwoChannelModel(num_samples=1000)
# Possible modes for data_model: 'Gaussian' or 'Parabola'
#y_1, y_2, Az_1, Az_2, z_1, z_2 = data_model('Gaussian', [1, 1, 0, 0, 0])

num_models = 3
lambda_reg = 1e-10
hdim = 256
#shared_dim = 5

for trial in range(0, 3):
    for t_ccor in [[1, 1, 0, 0, 0], [0.9, 0.9, 0, 0, 0], [0.7, 0.7, 0, 0, 0], [0.5, 0.5, 0, 0, 0],  [0.3, 0.3, 0, 0, 0], [0.1, 0.1, 0, 0, 0]]:
        for shared_dim in range(1, 5):
             try:
                 # Generate data
                 data_model = TwoChannelModel(num_samples=1000)
                 # Possible modes for data_model: 'Gaussian' or 'Parabola'
                 y_1, y_2, Az_1, Az_2, z_1, z_2 = data_model('Gaussian', t_ccor)
                 writer = nmca_model.create_grid_writer(LOGPATH, params=[trial, t_ccor, shared_dim])

                 optimizer = tf.keras.optimizers.Adam()
                 model = nmca_model.build_nmca_model(hdim)
                 for epoch in tqdm(range(100000), desc='Epochs'):
                     with tf.GradientTape() as tape:
                         # Watch the input to be able to compute the gradient later
                         tape.watch([y_1, y_2])
                         # Forward path
                         [fy_1, fy_2], [yhat_1, yhat_2] = model([tf.transpose(y_1), tf.transpose(y_2)])
                         # Loss computation
                         loss, cca_loss, rec_loss, ccor = nmca_model.compute_loss(y_1, y_2, fy_1, fy_2, yhat_1, yhat_2, shared_dim, lambda_reg=lambda_reg)

                         # Compute gradients
                         gradients = tape.gradient(loss, model.trainable_variables)
                         # Backpropagate through network
                         optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                         #print(gradients)
                         print('---')
                         for idx, m in enumerate(model.trainable_variables, 0):
                            print(f'{m.name} with shape {tf.shape(gradients[idx])}')
                         if epoch % 10 == 0:
                             B1, B2, epsilon, omega, ccor = CCA(fy_1, fy_2, 5)
                             if t_ccor[0] == 1:
                                 sim_v1 = nmca_model.compute_similarity_metric_v1(S=z_1[:2], U=0.5*(omega+epsilon))
                             else:
                                 sim_v1 = 0
                             sim_v2 = nmca_model.compute_similarity_metric_v1(S=z_1[:2], U=epsilon)
                             sim_v3 = nmca_model.compute_similarity_metric_v1(S=z_2[:2], U=omega)
                             dist = nmca_model.compute_distance_metric(S=z_1[:2], U=0.5 * (omega + epsilon)[:2])
                             #gradients_norm = tf.norm(gradients)
                             print('1')
                             for elem in gradients:
                                print(f'Gradients: {tf.shape(elem)}')
                             print('2')
                             write_gradients_summary(writer, gradients, model.trainable_variables)
                             write_metric_summary(writer, epoch, loss, cca_loss, rec_loss, ccor, dist, sim_v1, sim_v2, sim_v3)
                         if epoch % 2500 == 0:
                             #write_poly(writer, epoch, y_1, tf.transpose(yhat_1))
                             write_image_summary(writer, epoch, Az_1, Az_2, y_1, y_2, fy_1, fy_2)
                         if epoch % 5000 == 0:
                             write_PCC_summary(writer, epoch, z_1, z_2, epsilon, omega, 1000)

             except Exception as ex:
                 print(ex)