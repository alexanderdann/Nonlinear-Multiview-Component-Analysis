import tensorflow as tf
import numpy as np

class Unflatten3D(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(Unflatten3D, self).__init__()
        self.num_outputs = num_outputs

    def call(self, inputs):
        x = inputs.view(inputs.size()[0], [64, 4, 4])
        return x


class Flatten3D(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(Flatten3D, self).__init__()
        self.num_outputs = num_outputs

    def call(self, inputs):
        x_ind = tf.size(inputs)[0]
        x = tf.view(inputs, -1)
        #x = inputs.view(inputs.size()[0], -1)
        return x


class Unsqueeze(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(Unsqueeze, self).__init__()
        self.num_outputs = num_outputs

    def call(self, inputs):
        x = tf.expand_dims(inputs, -1)
        return x


class GradientReversal():
    def forward(self, inp):
        return inp

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return -grad_input

class NMCA():
    def __init__(self, input, structure, output):
        self.gamma_t = tf.Variable(0.1, tf.float32)
        self.dnn = [input] + structure + [output]
        self.layers = []
        self._build()

        tf.linalg.matmul(Z, W)
        P, _, Q = tf.linalg.svd()

    def lossfunction(self, theta_t, U_t):
        return 2

    def _build(self):
        for dim in self.dnn:
            self.layers.append(
                tf.keras.layers.Dense(dim, 'sigmoid')
            )
        self.model = tf.keras.Sequential(self.layers)

        self.model.compile(optimizer='adam',
                           loss=self.lossfunction,
                           )


class Decoder():
    def __init__(self, z_dim, c_dim, channels):
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


class CNNDAE():
    def __init__(self, input, structure, output):
        self.gamma_t = tf.Variable(0.1, tf.float32)
        self.dnn = [input] + structure + [output]
        self.layers = []
        self._build()

    def _build(self):
        for dim in self.dnn:
            self.layers.append(
                tf.keras.layers.Dense(dim, 'sigmoid')
            )
        self.model = tf.keras.Sequential(self.layers)

        self.model.compile(optimizer='adam',
                           loss=self.lossfunction,
                           )


if __name__ == '__main__':
    # Input data is of dimension M_q * N
    #model = NMCA(10, [10], 10)
    print(tf.version)