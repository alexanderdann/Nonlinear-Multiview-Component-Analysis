import os
import numpy as np
import tensorflow as tf
import scipy

from correlation_analysis import CCA


def _build_channel_model(inp, v_ind, ch_ind):
    # Build encoder
    enc_input = tf.keras.layers.Dense(
        256,
        activation='relu',
        name=f'View_{v_ind}_Encoder_DenseLayer_Channel_{ch_ind}'
    )(inp)
    enc_output = tf.keras.layers.Dense(
        1,
        activation=None,
        name=f'View_{v_ind}_Encoder_OutputLayer_Channel_{ch_ind}'
    )(enc_input)

    # Build decoder
    x = tf.keras.layers.Dense(
        256,
        activation='relu',
        name=f'View_{v_ind}_Decoder_DenseLayer_Channel_{ch_ind}'
    )(enc_output)
    dec_output = tf.keras.layers.Dense(
        1,
        activation=None,
        name=f'View_{v_ind}_Decoder_OutputLayer_Channel_{ch_ind}'
    )(x)

    return enc_output, dec_output


def _build_view_models(inputs, view_ind):
    enc_outputs = list()
    dec_outputs = list()

    for i in range(5):
        enc_output, dec_output = _build_channel_model(inputs[i], view_ind, i)
        enc_outputs.append(enc_output)
        dec_outputs.append(dec_output)

    enc_outputs = tf.keras.layers.concatenate(
        enc_outputs,
        name=f'View_{view_ind}_Encoder_OutputConcatenation'
    )
    dec_outputs = tf.keras.layers.concatenate(
        dec_outputs,
        name=f'View_{view_ind}_Decoder_OutputConcatenation'
    )

    return enc_outputs, dec_outputs


def build_nmca_model():
    inp_view_1 = tf.keras.layers.Input(shape=(5))
    inp_view_2 = tf.keras.layers.Input(shape=(5))
    view_1_splits = tf.split(inp_view_1, num_or_size_splits=5, axis=1)
    view_2_splits = tf.split(inp_view_2, num_or_size_splits=5, axis=1)

    enc_outputs_1, dec_outputs_1 = _build_view_models(view_1_splits, 0)
    enc_outputs_2, dec_outputs_2 = _build_view_models(view_2_splits, 1)

    model = tf.keras.Model(
        inputs=[inp_view_1, inp_view_2],
        outputs=[[enc_outputs_1, enc_outputs_2], [dec_outputs_1, dec_outputs_2]]
    )

    return model


def compute_loss(y_1, y_2, fy_1, fy_2, yhat_1, yhat_2, lambda_reg=0.01):
    # CCA loss
    B1, B2, epsilon, omega, ccor = CCA(fy_1, fy_2, 2)
    cca_loss = tf.reduce_mean(tf.square(tf.norm(tf.subtract(epsilon, omega), axis=0)))

    # Reconstruction loss
    rec_loss_1 = tf.square(tf.norm(tf.transpose(y_1) - yhat_1, axis=0))
    rec_loss_2 = tf.square(tf.norm(tf.transpose(y_2) - yhat_2, axis=0))
    rec_loss = tf.reduce_mean(tf.add(rec_loss_1, rec_loss_2))

    # Combine losses
    loss = cca_loss + lambda_reg * rec_loss

    return loss, cca_loss, rec_loss, ccor

def compute_distance_metric(S, U):
    Ps = np.eye(S.shape[1]) - tf.transpose(S)@np.linalg.inv(S@tf.transpose(S))@S
    Q = scipy.linalg.orth(tf.transpose(U))
    dist = np.linalg.norm(Ps@Q, ord=2)
    return dist

def create_writer(root_dir):
    folders = list()
    for file in os.listdir(root_dir):
        file_path = os.path.join(root_dir, file)
        if os.path.isdir(file_path):
            folders.append(file_path)

    curr_number = 0
    while True:
        num_str = str(curr_number)
        if len(num_str) == 1:
            num_str = "0"+num_str

        folder = os.path.join(root_dir, num_str)
        if not os.path.exists(folder):
            break
        else:
            curr_number = curr_number + 1

    os.makedirs(folder)

    return tf.summary.create_file_writer(folder)