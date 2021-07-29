import tensorflow as tf
import numpy as np
from CustomizedLayers import GradientReversal, Flatten3D, Unflatten3D


class CCA():
    def __init__(self, view1, view2, outdim):
        self.A, self.B, self.epsilon, self.omega, self.ccor = self._calculate(view1, view2, outdim)

    def getitems(self):
        return (self.A, self.B), (self.epsilon, self.omega, self.ccor)

    def _calculate(self, view1, view2, outdim):
        V1 = tf.Variable(view1, dtype=tf.float64)
        V2 = tf.Variable(view2, dtype=tf.float64)

        r1 = 1e-2
        r2 = 1e-2

        assert V1.shape[0] == V2.shape[0]
        M = tf.constant(V1.shape[0], dtype=tf.float64)
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
        D, U, V = tf.linalg.svd(C, full_matrices=True)

        print(f'U: {U.shape} S: {Sigma11_root_inv.shape}')

        A = tf.tensordot(tf.transpose(U), Sigma11_root_inv, axes=1)
        B = tf.tensordot(tf.transpose(V), Sigma22_root_inv, axes=1)

        epsilon = tf.tensordot(A, tf.transpose(V1_bar), axes=1)
        omega = tf.tensordot(B, tf.transpose(V2_bar), axes=1)

        est_X = epsilon[:, 0:outdim]
        est_Y = omega[:, 0:outdim]

        print("Canonical Correlations: " + str(D[0:outdim]))

        return A, B, epsilon, omega, D[0:outdim]


class NonlinearComponentAnalysis(tf.keras.Model):
    def __init__(self, num_views, z_dim, c_dim, encoder_layers, decoder_layers):
        super(NonlinearComponentAnalysis, self).__init__()
        self.num_views = num_views
        self.gamma_t = tf.Variable(0.1, tf.float32)

        self.model = self._build(z_dim, c_dim, encoder_layers, decoder_layers)

    def _build(self, z_dim, c_dim, encoder_layers, decoder_layers):
        model_view1 = tf.keras.Sequential(
            tf.keras.layers.InputLayer(input_dim=1)
        )

        for encoder_info in encoder_layers:
            model_view1.add(
                tf.keras.layers.Dense(encoder_info[0], activation=encoder_info[1])
            )

        model_view2 = tf.keras.Sequential(
            tf.keras.layers.InputLayer(input_dim=1)
        )

        for encoder_info in encoder_layers:
            model_view2.add(
                tf.keras.layers.Dense(encoder_info[0], activation=encoder_info[1])
            )

        # Get intermediate representations to calculate CCA
        self.weights1 = model_view1.get_layer(index=-1)
        self.weights2 = model_view1.get_layer(index=-1)
        # returns (A, B) (epsilon, omega, canonical_correlations)
        # Where A and B are the transformation matrices to achieve
        # canonical variables epsilon and omega, as well as the
        # canonical correlations
        t_matrices, _ = CCA(self.weights1, self.weights2, z_dim + c_dim).getitems()

        # To keep notation as in the paper
        self.B_1 = t_matrices[0]
        self.B_2 = t_matrices[1]

        for decoder_info in decoder_layers:
            model_view1.add(
                tf.keras.layers.Dense(decoder_info[0], activation=decoder_info[1])
            )

        for decoder_info in decoder_layers:
            model_view2.add(
                tf.keras.layers.Dense(decoder_info[0], activation=decoder_info[1])
            )

   

    def encode(self, input):
        shared_components, private_components = [], []
        for view in range(self.num_views):
            shared, private = self.Encoders[view](input[view])
            shared_components.append(shared)
            private_components.append(private)

        return shared_components, private_components

    def decode(self, shared, private):
        reconstructed = []
        for view in range(self.num_views):
            tmp = self.Decoders[view](shared[view], private[view])
            reconstructed.append(tmp)

        return reconstructed

    def loss(self, final1, final2):
        init1 = self.inital1
        init2 = self.inital2

        U = self.update_U()

        lambda_reg = tf.constant(0.1, dtype=tf.float64)
        arg1 = tf.Variable(U - tf.matmul(self.B_1, self.weights1), dtype=tf.float64)
        arg2 = tf.Variable(U - tf.matmul(self.B_2, self.weights2), dtype=tf.float64)
        args = [arg1, arg2]
        tmp_loss1 = [tf.math.reduce_euclidean_norm(arg) ** 2 for arg in args]
        tmp_loss1 = tf.math.reduce_sum(tf.Variable(tmp_loss1, dtype=tf.float64))

        reg_args = [tf.Variable(init1 - final1, dtype=tf.float64),
                    tf.Variable(init2 - final2, dtype=tf.float64)]
        tmp_loss2 = [tf.math.reduce_euclidean_norm(arg) ** 2 for arg in reg_args]
        tmp_loss2 = lambda_reg * tf.math.reduce_sum(tf.Variable(tmp_loss2, dtype=tf.float64))

        final_loss = tmp_loss1 + tmp_loss2
        return final_loss