import os
import tensorflow as tf
import numpy as np
import scipy
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import nmca_model
from TwoChannelModel import TwoChannelModel
from correlation_analysis import CCA, PCC_Matrix
from plot import plot_eval
from tf_summary import write_image_summary, write_metric_summary


LOGPATH = ""


# Generate data
data_model = TwoChannelModel(num_samples=1000)
y_1, y_2, Az_1, Az_2, z_1, z_2 = data_model()

num_models = 3
lambda_reg = 1e-10

for _ in range(5):
    writer = nmca_model.create_writer(LOGPATH)
    optimizer = tf.keras.optimizers.Adam()
    model = nmca_model.build_nmca_model()

    for epoch in tqdm(range(10000), desc='Epochs'):
        with tf.GradientTape() as tape:
            # Watch the input to be able to compute the gradient later
            tape.watch([y_1,y_2])
            # Forward path
            [fy_1, fy_2], [yhat_1, yhat_2] = model([tf.transpose(y_1), tf.transpose(y_2)])
            # Loss computation
            loss, cca_loss, rec_loss, ccor = nmca_model.compute_loss(y_1, y_2, fy_1, fy_2, yhat_1, yhat_2, lambda_reg=lambda_reg)

            if epoch%5 == 0:
                # Compute dist metric
                B1, B2, epsilon, omega, ccor = CCA(fy_1, fy_2, 2)
                dist = nmca_model.compute_distance_metric(S=z_1[:2], U=0.5*(omega+epsilon))

                write_metric_summary(writer, epoch, loss, cca_loss, rec_loss, ccor, dist)

            if epoch%500 == 0:
                write_image_summary(writer, epoch, Az_1, Az_2, y_1, y_2, fy_1, fy_2)

        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        # Backpropagate through network
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))