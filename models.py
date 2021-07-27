import tensorflow as tf
import numpy as np
from CustomizedLayers import GradientReversal, GradientReversalOperator, Flatten3D, Unflatten3D, Unsqueeze

class MLP(tf.keras.Model):
    def __init__(self, input_dim, hidden_layers, output_dim, probability=0.0):
        super(MLP, self).__init__()
        self.network = tf.keras.Sequential(tf.keras.layers.InputLayer(input_dim))
        structure = hidden_layers + [output_dim]

        for dim in structure:
            self.network.add(tf.keras.layers.Dense(dim, activation='relu'))
            # Reread paper with regards to dropout
            self.network.add(tf.keras.layers.Dropout(dim))

    def call(self, input):
        result = self.network(input)
        return result

class MiniMaxCCA(tf.keras.Model):
    def __init__(self, view1_dim, view2_dim, phi_size, tau_size, latent_dim=1):
        super(MiniMaxCCA, self).__init__()
        self.phi = MLP(view1_dim, phi_size, latent_dim)
        self.tau = MLP(view2_dim, tau_size, latent_dim)

        self.GradReverse1 = GradientReversal()
        self.GradReverse2 = GradientReversal()

    def call(self, input):
        phi_reg = self.phi(self.GradReverse1(input))
        tau_reg = self.tau(self.GradReverse2(input))

        return phi_reg, tau_reg


class CNNencoder(tf.keras.Model):
    def __init__(self, z_dim, c_dim, channels):
        super(CNNencoder, self).__init__()
        self.model, self.S, self.P = self._build(z_dim, c_dim, channels)

    def _build(self, z_dim, c_dim, channels):
        model = tf.keras.Sequential(
            tf.keras.layers.InputLayer(32),
            tf.keras.layers.Conv2DTranspose(channels, 4, 2, 'valid', activation='relu'),
            tf.keras.layers.Conv2DTranspose(32, 4, 2, 'valid', activation='relu'),
            tf.keras.layers.Conv2DTranspose(64, 4, 2, 'valid', activation='relu'),
            tf.keras.layers.Conv2DTranspose(64, 4, 2, 'valid', activation='relu'),
            Flatten3D(64),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu')
        )

        S = tf.keras.Sequential(
            tf.keras.layers.InputLayer(256),
            tf.keras.layers.Dense(z_dim)
        )

        P = tf.keras.Sequential(
            tf.keras.layers.InputLayer(256),
            tf.keras.layers.Dense(c_dim)
        )

        return model, S, P

    def call(self, input):
        combined_view = self.model(input)
        shared_view = self.S(combined_view)
        private_view = self.P(combined_view)
        return shared_view, private_view






class CNNdecoder(tf.keras.Model):
    def __init__(self, z_dim, c_dim, channels):
        super(CNNdecoder, self).__init__()
        self.model = self._build(z_dim, c_dim, channels)

    def _build(self, z_dim, c_dim, channels):
        model = tf.keras.Sequential(
            tf.keras.layers.Flatten(input_dim=z_dim + c_dim),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1024, activation='relu'),
            Unflatten3D(64),
            tf.keras.layers.Conv2DTranspose(64, 4, 2, 'valid', activation='relu'),
            tf.keras.layers.Conv2DTranspose(32, 4, 2, 'valid', activation='relu'),
            tf.keras.layers.Conv2DTranspose(32, 4, 2, 'valid', activation='relu'),
            tf.keras.layers.Conv2DTranspose(channels, 4, 2, 'valid', activation='relu')
            )
        return model

    def call(self, shared, private):
        input = tf.concat([shared, private], 1)
        output = self.model(input)
        return output





class CNNDAE(tf.keras.Model):
    def __init__(self, num_views, z_dim=10, c_dim=2, channels=1):
        super(CNNDAE, self).__init__()
        self.num_views = num_views
        self.gamma_t = tf.Variable(0.1, tf.float32)

        self.Encoders, self.Decoders = [], []
        for view in range(self.num_views):
            self.Encoders.append(CNNencoder(z_dim, c_dim, channels))
            self.Decoders.append(CNNdecoder(z_dim, c_dim, channels))

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

    def call(self, input):
        shared_components, private_components = self.encode(input)
        reconstructed = self.decode(shared_components, private_components)
        return shared_components, private_components, reconstructed

class CCA():
    def __init__(self, view1, view2, outdim):
        self._calculate(view1, view2, outdim)


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

        Sigma12 = (tf.linalg.matmul(tf.transpose(V1_bar), V2_bar)) / (M-1)
        Sigma11 = (tf.linalg.matmul(tf.transpose(V1_bar), V1_bar) + r1 * np.eye(V1.shape[1])) / (M-1)
        Sigma22 = (tf.linalg.matmul(tf.transpose(V2_bar), V2_bar) + r2 * np.eye(V2.shape[1])) / (M-1)

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





