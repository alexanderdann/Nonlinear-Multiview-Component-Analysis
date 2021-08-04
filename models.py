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
        V1 = tf.constant(view1, dtype=tf.float64)
        V2 = tf.constant(view2, dtype=tf.float64)

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

        print("Canonical Correlations: " + str(D))

        return A, B, epsilon, omega, D


class NonlinearComponentAnalysis(tf.keras.Model):
    def __init__(self, num_views, encoder_layers, decoder_layers):
        super(NonlinearComponentAnalysis, self).__init__()
        self.num_views = num_views
        self.gamma_t = tf.Variable(0.1, tf.float32)
        self.U = 0
        self.optimizer1 = tf.keras.optimizers.Adam(learning_rate=self.gamma_t)
        self.optimizer2 = tf.keras.optimizers.Adam(learning_rate=self.gamma_t)

        self.model1, self.model2 = self._build(encoder_layers, decoder_layers)

        print('Input Layer is there, but TF does not show it in the summary.\n')
        print('Summaries of the models are following...')
        self.model1.summary()
        self.model2.summary()

        self.model1.compile(optimizer='adam')
        self.model2.compile(optimizer='adam')

    def _build(self, encoder_layers, decoder_layers):
        model_view1 = tf.keras.Sequential(
            tf.keras.layers.InputLayer(input_shape=(encoder_layers[0][0],), name='Input_Layer_Model_1'),
            name='First_Model'
        )

        # self.enc_name1 will automatically save the name for the last layer of the
        # Encoder -> Needed for CCA
        for counter, encoder_info in enumerate(encoder_layers[1:], 1):
            self.enc_name1 = f'M1_Encoder_{counter}'
            print(encoder_info[0])
            model_view1.add(
                tf.keras.layers.Dense(encoder_info[0], activation=encoder_info[1], name=self.enc_name1)
            )

        model_view2 = tf.keras.Sequential(
            tf.keras.layers.InputLayer(input_shape=(encoder_layers[0][0],), name='Input_Layer_Model_2'),
            name='Second Model'
        )

        # self.enc_name2 will automatically save the name for the last layer of the
        # Encoder -> Needed for CCA
        for counter, encoder_info in enumerate(encoder_layers[1:], 1):
            self.enc_name2 = f'M2_Encoder_{counter}'
            model_view2.add(
                tf.keras.layers.Dense(encoder_info[0], activation=encoder_info[1], name=self.enc_name2)
            )


        for counter, decoder_info in enumerate(decoder_layers, 1):
            name = f'M1_Decoder_{counter}'
            model_view1.add(
                tf.keras.layers.Dense(decoder_info[0], activation=decoder_info[1], name=name)
            )

        for counter, decoder_info in enumerate(decoder_layers, 1):
            name = f'M2_Decoder_{counter}'
            model_view2.add(
                tf.keras.layers.Dense(decoder_info[0], activation=decoder_info[1], name=name)
            )

        return model_view1, model_view2


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

    def get_B(self):
        # Get intermediate representations to calculate CCA
        self.layer1 = self.model1.get_layer(name=self.enc_name1).get_weights()[0]
        self.layer2 = self.model2.get_layer(name=self.enc_name2).get_weights()[0]

        #self.layer1 = self.model1.get_layer(name=self.enc_name1).output
        #self.layer2 = self.model2.get_layer(name=self.enc_name2).output
        # returns (A, B) (epsilon, omega, canonical_correlations)
        # Where A and B are the transformation matrices to achieve
        # canonical variables epsilon and omega, as well as the
        # canonical correlations
        t_matrices, self.trash = CCA(self.layer1, self.layer2).getitems()

        # To keep notation similar as in the paper
        B_1 = t_matrices[0]
        B_2 = t_matrices[1]

        return B_1, B_2

    def update_U(self, B_views, N):
        dim = B_views[0].shape[1]
        print(dim)
        W = tf.eye(dim, dim) - tf.matmul(tf.ones([dim, dim]), tf.transpose(tf.ones([dim, dim])))
        print(f'W: {W}')
        print(f'B: {B_views[0]}')
        int_U = tf.reduce_sum(tf.constant([tf.matmul(B, W) for B in B_views]))
        P, D, Q = tf.linalg.svd(int_U)
        return tf.Variable(tf.sqrt(N)*tf.matmul(P, tf.transpose(Q)))

    def loss(self, final1, final2, init1, init2):
        B_1, B_2 = self.get_B()
        self.U = self.update_U([B_1, B_2], 1)

        lambda_reg = tf.constant(0.1, dtype=tf.float64)
        arg1 = tf.Variable(self.U - tf.matmul(self.B_1, self.weights1), dtype=tf.float64)
        arg2 = tf.Variable(self.U - tf.matmul(self.B_2, self.weights2), dtype=tf.float64)
        args = [arg1, arg2]
        tmp_loss1 = [tf.math.reduce_euclidean_norm(arg) ** 2 for arg in args]
        tmp_loss1 = tf.math.reduce_sum(tf.Variable(tmp_loss1, dtype=tf.float64))

        reg_args = [tf.Variable(init1 - final1, dtype=tf.float64),
                    tf.Variable(init2 - final2, dtype=tf.float64)]
        tmp_loss2 = [tf.math.reduce_euclidean_norm(arg) ** 2 for arg in reg_args]
        tmp_loss2 = lambda_reg * tf.math.reduce_sum(tf.Variable(tmp_loss2, dtype=tf.float64))

        final_loss = tmp_loss1 + tmp_loss2
        return final_loss