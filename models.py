import tensorflow as tf
import numpy as np


def BatchPreparation(batch_size, samples, data):
    assert data.shape[1] == samples
    assert samples % batch_size == 0

    batched_data = []
    for ind in range(samples//batch_size):
        batched_data.append(data[:, ind*batch_size: (ind+1)*batch_size])

    return tf.constant(batched_data, name='Batched Data')

class CCA():
    def __init__(self, view1, view2, shared_dim):
        self.A, self.B, self.epsilon, self.omega, self.ccor = self._calculate(view1, view2, shared_dim)

    def getitems(self):
        return (self.A, self.B), (self.epsilon, self.omega, self.ccor)

    def _calculate(self, view1, view2, shared_dim):
        print('\n\n\n--------- CCA Start ---------\n\n')

        V1 = tf.cast(view1, dtype=tf.float32)
        V2 = tf.cast(view2, dtype=tf.float32)

        r1 = 0.0
        r2 = 0.0

        assert V1.shape[0] == V2.shape[0]
        M = tf.constant(V1.shape[0], dtype=tf.float32)
        ddim = tf.constant(V1.shape[1], dtype=tf.int16)
        # check mean and variance

        mean_V1 = tf.reduce_mean(V1, 0)
        mean_V2 = tf.reduce_mean(V2, 0)

        V1_bar = tf.subtract(V1, tf.tile(tf.convert_to_tensor(mean_V1)[None], [M, 1]))
        #print(tf.tile(tf.convert_to_tensor(mean_V1)[None], [M, 1]))
        V2_bar = tf.subtract(V2, tf.tile(tf.convert_to_tensor(mean_V2)[None], [M, 1]))

        Sigma12 = tf.linalg.matmul(tf.transpose(V1_bar), V2_bar) / (M - 1)
        Sigma11 = tf.linalg.matmul(tf.transpose(V1_bar), V1_bar) / (M - 1) + r1 * tf.eye(ddim)
        Sigma22 = tf.linalg.matmul(tf.transpose(V2_bar), V2_bar) / (M - 1) + r2 * tf.eye(ddim)

        Sigma11_root_inv = tf.linalg.sqrtm(tf.linalg.inv(Sigma11))
        Sigma22_root_inv = tf.linalg.sqrtm(tf.linalg.inv(Sigma22))
        Sigma22_root_inv_T = tf.transpose(Sigma22_root_inv)

        C = tf.linalg.matmul(tf.linalg.matmul(Sigma11_root_inv, Sigma12), Sigma22_root_inv_T)
        #print(f'C Shape: {C.shape}\n\nC values: {C}\n')
        D, U, V = tf.linalg.svd(C, full_matrices=True)

        A = tf.matmul(tf.transpose(U)[:shared_dim], Sigma11_root_inv)
        B = tf.matmul(tf.transpose(V)[:shared_dim], Sigma22_root_inv)

        epsilon = tf.matmul(A, tf.transpose(V1_bar))
        omega = tf.matmul(B, tf.transpose(V2_bar))

        print("Canonical Correlations: " + str(D))
        print('\n\n--------- CCA End ---------')

        #print(f'B DIM {tf.shape(epsilon)}')

        return A, B, epsilon, omega, D


class NonlinearComponentAnalysis(tf.keras.Model):
    def __init__(self, num_views, num_channels, encoder_layers, decoder_layers, batch_size):
        super(NonlinearComponentAnalysis, self).__init__()
        self.num_views = num_views
        self.gamma_t = tf.Variable(10**-3, tf.float32)
        self.U = 0
        self.optimizer = tf.keras.optimizers.Adam()
        self.mse = tf.keras.losses.MeanSquaredError()
        self.can_corr = []

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
        model.compile()

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

    def getCCA(self, est_view1, est_view2, shared_dim):
        # returns (A, B) (epsilon, omega, canonical_correlations)
        # Where A and B are the transformation matrices to achieve
        # canonical variables epsilon and omega, as well as the
        # canonical correlations
        t_matrices, cca_data = CCA(est_view1, est_view2, shared_dim).getitems()

        # To keep notation similar as in the paper
        B_1 = t_matrices[0]
        B_2 = t_matrices[1]

        self.est_sources = (cca_data[0], cca_data[1])

        self.can_corr.append(cca_data[2])


        return B_1, B_2, cca_data[0], cca_data[1]

    def update_U(self, B_views, batch_size, encoder_data):
        #print(f'\nB Shape: {tf.shape(B_views)}')
        #print(f'Encoder Shape: {tf.shape(encoder_data[0])}\n')

        dim = encoder_data[0].shape[0]
        n_views = tf.shape(B_views)[0]
        half = tf.constant(0.5, dtype=tf.float32)
        I_t = tf.cast(batch_size, dtype=tf.float32)
        W = tf.eye(dim, dim) - tf.matmul(tf.ones([dim, dim]), tf.transpose(tf.ones([dim, dim])))/I_t

        assert n_views == 2
        int_Z = [half*tf.matmul(B_views[i], tf.transpose(encoder_data[i])) for i in range(n_views)]
        Z = tf.add(int_Z[0], int_Z[1])

        int_U = tf.matmul(Z, W)

        D, P, Q = tf.linalg.svd(int_U, full_matrices=False)

        # singular values - left singular vectors - right singular vectors
        # print(f'{tf.shape(D)} - {tf.shape(P)} - {tf.shape(Q)}')

        return tf.sqrt(I_t)*tf.matmul(P, tf.transpose(Q))


    def loss(self, enc1, enc2, dec1, dec2, init1, init2, batch_size, shared_dim):
        input1 = tf.cast(init1, dtype=tf.float32)
        input2 = tf.cast(init2, dtype=tf.float32)

        encoder1 = tf.cast(enc1, dtype=tf.float32)
        encoder2 = tf.cast(enc2, dtype=tf.float32)

        decoder1 = tf.cast(dec1, dtype=tf.float32)
        decoder2 = tf.cast(dec2, dtype=tf.float32)

        B_1, B_2, epsilon, omega = self.getCCA(encoder1, encoder2, shared_dim)
        #self.U = self.update_U([B_1, B_2], batch_size, [encoder1, encoder2])
        #self.U = self.update_U_2(shared_dim, batch_size)
        Z = tf.subtract(epsilon, omega)

        lambda_reg = tf.constant(0.6, dtype=tf.float32)

        #arg1 = Z - tf.matmul(B_1, tf.transpose(encoder1))
        #arg2 = Z - tf.matmul(B_2, tf.transpose(encoder2))
        #args = [arg1, arg2]
        loss1 = tf.math.reduce_euclidean_norm(Z) ** 2

        reg_args = [tf.subtract(input1, decoder1),
                    tf.subtract(input2, decoder2)]
        tmp_loss2 = [tf.math.reduce_euclidean_norm(arg) ** 2 for arg in reg_args]
        loss2 = lambda_reg * tf.math.reduce_sum(tmp_loss2)

        final_loss = loss1 + loss2

        print(f'\n######## Loss: {final_loss} ########\n')
        return final_loss
