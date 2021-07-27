import tensorflow as tf

@tf.custom_gradient
def GradientReversalOperator(x):
    def grad(dy):
        return -1 * dy

    return x, grad

class GradientReversal(tf.keras.layers.Layer):
    def __init__(self):
        super(GradientReversal, self).__init__()

    def call(self, inputs):
        return GradientReversalOperator(inputs)


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
