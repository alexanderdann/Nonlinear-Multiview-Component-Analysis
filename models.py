import tensorflow as tf
import visualkeras
import numpy as np
from CustomizedLayers import GradientReversal, Flatten3D, Unflatten3D

def BatchPreparation(batch_size, samples, data):
    assert data.shape[0] == samples

    batched_data = []
    data_length = data.shape[0]
    for ind in range(data_length//batch_size):
        batched_data.append(data[ind*batch_size: (ind+1)*batch_size])

    return tf.constant(batched_data, name='Batched Data')

class CCA():
    def __init__(self, view1, view2):
        self.A, self.B, self.epsilon, self.omega, self.ccor = self._calculate(view1, view2)

    def getitems(self):
        return (self.A, self.B), (self.epsilon, self.omega, self.ccor)

    def _calculate(self, view1, view2):
        V1 = tf.constant(view1, dtype=tf.float32)
        V2 = tf.constant(view2, dtype=tf.float32)

        r1 = 1e-2
        r2 = 1e-2

        print(f'V1: {V1.shape}')

        assert V1.shape[0] == V2.shape[0]
        M = tf.constant(V1.shape[0], dtype=tf.float32)
        meanV1 = tf.reduce_mean(V1, axis=0, keepdims=True)
        meanV2 = tf.reduce_mean(V2, axis=0, keepdims=True)

        V1_bar = V1 - tf.tile(meanV1, [M, 1])
        V2_bar = V2 - tf.tile(meanV2, [M, 1])

        print(f'V1_Bar: {V1_bar.shape}')

        Sigma12 = (tf.linalg.matmul(tf.transpose(V1_bar), V2_bar)) / (M - 1)
        Sigma11 = (tf.linalg.matmul(tf.transpose(V1_bar), V1_bar) + r1 * np.eye(V1.shape[1])) / (M - 1)
        Sigma22 = (tf.linalg.matmul(tf.transpose(V2_bar), V2_bar) + r2 * np.eye(V2.shape[1])) / (M - 1)

        Sigma11_root_inv = tf.linalg.sqrtm(tf.linalg.inv(Sigma11))
        Sigma22_root_inv = tf.linalg.sqrtm(tf.linalg.inv(Sigma22))
        Sigma22_root_inv_T = tf.transpose(Sigma22_root_inv)

        C = tf.linalg.matmul(tf.linalg.matmul(Sigma11_root_inv, Sigma12), Sigma22_root_inv_T)
        print(f'C Shape: {C.shape}\nC val: {C}')
        D, U, V = tf.linalg.svd(C, full_matrices=True)

        print(f'U: {U.shape} S: {Sigma11_root_inv.shape}')

        A = tf.tensordot(tf.transpose(U), Sigma11_root_inv, axes=1)
        B = tf.tensordot(tf.transpose(V), Sigma22_root_inv, axes=1)

        epsilon = tf.tensordot(A, tf.transpose(V1_bar), axes=1)
        omega = tf.tensordot(B, tf.transpose(V2_bar), axes=1)

        print("Canonical Correlations: " + str(D))

        return A, B, epsilon, omega, D


class NonlinearComponentAnalysis(tf.keras.Model):
    def __init__(self, num_views, num_channels, encoder_layers, decoder_layers, batch_size):
        super(NonlinearComponentAnalysis, self).__init__()
        self.num_views = num_views
        self.gamma_t = tf.Variable(10**-3, tf.float32)
        self.U = 0
        self.optimizer1 = tf.keras.optimizers.Adam(learning_rate=self.gamma_t)
        self.optimizer2 = tf.keras.optimizers.Adam(learning_rate=self.gamma_t)

        # Implementation only for 2 views for now
        assert num_views == 2

        self.NCA = self._connect(encoder_layers, decoder_layers, num_views, num_channels, batch_size)

    def _connect(self, encoder_layers, decoder_layers, num_views, num_channels, batch_size):
        # Idea: encapsulate each channel specific en-/decoder
        # into broader encoder and decoder structure, which consists
        # of smaller parts.
        #
        # Benefit: easier application and flexibility
        #
        # Solution: for each view, we stack all channel specific en-/decoders
        # in self.channel_encoders/self.channel_decoders, this list is
        # view specific and passed after N=num(channels) to the
        # self.encoders/self.decoders.
        #
        # This means, self.encoders is of dim num_views x num_channels
        # e.g. self.encoders = [
        #                       [encoder_view1_channel1, encoder_view1_channel2],
        #                       [encoder_view2_channel1, encoder_view2_channel2]
        #                       ]
        self.encoders_outs = []
        self.decoders_outs = []
        self.view_input = []

        for view in range(num_views):
            self.channel_encoders_out, self.channel_decoders_out = [], []

            for channel in range(num_channels):
                #name = f'View_{view}_Channel_{channel}'
                input, outputs = self._buildSeq(encoder_layers, decoder_layers, channel, view, batch_size)
                self.view_input.append(input)
                self.channel_encoders_out.append(outputs[0])
                self.channel_decoders_out.append(outputs[1])

            self.decoders_outs.append(
                tf.keras.layers.concatenate(self.channel_decoders_out, name=f'View_{view}_Final_Layer')
            )
            self.encoders_outs.append(
                tf.keras.layers.concatenate(self.channel_encoders_out, name=f'View_{view}_Pre_CCA_Layer')
            )

        model = tf.keras.Model(inputs=self.view_input, outputs=[self.encoders_outs, self.decoders_outs])

        print(f'Visualization following...\n')
        tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

        print(f'Summary following...\n')
        model.summary()

        custom_loss = self.loss(self.encoders_outs[0], self.encoders_outs[1],
                                self.decoders_outs[0], self.decoders_outs[1],
                                self.view_input[0], self.view_input[1],
                                )
        model.add_loss(custom_loss)
        model.compile(optimizer='adam', run_eagerly=True)

        return model

    def _buildSeq(self, encoder_layers, decoder_layers, ch_ind, v_ind, batch_size):
        initial_input_enc = tf.keras.Input(shape=(batch_size, encoder_layers[0][0]),
                                           name=f'View_{v_ind}_Input_Encoder_Layer_Channel_{ch_ind}')
        final_output_enc = 0
        for counter, encoder_info in enumerate(encoder_layers[1:], 0):
            name = f'View_{v_ind}_Encoder_Layer_{counter}_Channel_{ch_ind}'
            if counter == 0:
                # first iteration and temporary input equals initial input
                input = tf.keras.layers.Dense(
                    encoder_info[0], activation=encoder_info[1], name=name)(initial_input_enc)
            else:
                input = tf.keras.layers.Dense(
                    encoder_info[0], activation=encoder_info[1], name=name)(input)

                final_output_enc = input

        #initial_input_dec = tf.keras.Input(shape=(decoder_layers[0][0]),
        #                                   name=f'View_{v_ind}_Input_Decoder_Layer_Channel_{ch_ind}')
        final_output_dec = 0
        for counter, decoder_info in enumerate(decoder_layers[1:], 0):
            name = f'View_{v_ind}_Decoder_Layer_{counter}_Channel_{ch_ind}'
            if counter == 0:
                # first iteration and temporary input equals initial input
                name = f'View_{v_ind}_Input_Decoder_Layer_Channel_{ch_ind}'
                input = tf.keras.layers.Dense(
                    decoder_info[0], activation=decoder_info[1], name=name)(final_output_enc)
            else:
                input = tf.keras.layers.Dense(
                    decoder_info[0], activation=decoder_info[1], name=name)(input)

                final_output_dec = input

        return initial_input_enc, (final_output_enc, final_output_dec)

    def get_B(self, est_view1, est_view2):
        # returns (A, B) (epsilon, omega, canonical_correlations)
        # Where A and B are the transformation matrices to achieve
        # canonical variables epsilon and omega, as well as the
        # canonical correlations
        t_matrices, _ = CCA(est_view1, est_view2).getitems()

        # To keep notation similar as in the paper
        B_1 = t_matrices[0]
        B_2 = t_matrices[1]

        print(f'---- {B_1} -----')

        return B_1, B_2

    def update_U(self, B_views, N):
        dim = B_views[0].shape[1]
        N = 1024
        print(N)
        # In this case now dim = 1 what means that W will be zero
        W = tf.eye(dim, dim) - tf.matmul(tf.ones([dim, dim]), tf.transpose(tf.ones([dim, dim])))/N
        print(f'W: {W[0]}')
        print(f'B: {B_views[0][0]}')
        try:
            # If B and W are matrices/vectors
            int_U = tf.reduce_sum(tf.constant([tf.matmul(B, W) for B in B_views]))
        except:
            # If B and W are scalars
            int_U = tf.reduce_sum([tf.tensordot(B[0], W[0], axes=0) for B in B_views])

        print(B_views[0])
        print(B_views[1])
        print(f'Intermediate U: {int_U}\n')

        P, D, Q = tf.linalg.svd(int_U)
        return tf.Variable(tf.sqrt(N)*tf.matmul(P, tf.transpose(Q)))

    def loss(self, enc1, enc2, dec1, dec2, init1, init2):
        def wrapped_loss():
            print('HELLO')
            print(enc1)
            print('bye')
            B_1, B_2 = self.get_B(enc1, enc2)
            self.U = self.update_U([B_1, B_2], 1)

            lambda_reg = tf.constant(0.1, dtype=tf.float64)
            arg1 = tf.Variable(self.U - tf.matmul(self.B_1, self.weights1), dtype=tf.float64)
            arg2 = tf.Variable(self.U - tf.matmul(self.B_2, self.weights2), dtype=tf.float64)
            args = [arg1, arg2]
            tmp_loss1 = [tf.math.reduce_euclidean_norm(arg) ** 2 for arg in args]
            tmp_loss1 = tf.math.reduce_sum(tf.Variable(tmp_loss1, dtype=tf.float64))

            reg_args = [tf.Variable(init1 - dec1, dtype=tf.float64),
                        tf.Variable(init2 - dec2, dtype=tf.float64)]
            tmp_loss2 = [tf.math.reduce_euclidean_norm(arg) ** 2 for arg in reg_args]
            tmp_loss2 = lambda_reg * tf.math.reduce_sum(tf.Variable(tmp_loss2, dtype=tf.float64))

            final_loss = tmp_loss1 + tmp_loss2
            return final_loss
        return wrapped_loss
