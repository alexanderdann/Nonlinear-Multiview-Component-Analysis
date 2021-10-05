import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def _sigmoid(x):
    return 1/(1 + np.exp(-x))


class TwoChannelModel():
    def __init__(self, num_samples):
        np.random.seed(3333)
        self.num_samples = num_samples

    def __call__(self):
        return self._generate_samples()

    def _generate_samples(self):
        # Mixing matrices
        A_1 = np.random.normal(size=(5, 5))
        A_2 = np.random.normal(size=(5, 5))

        # Shared parabola data
        s1 = np.linspace(-1, 1, self.num_samples)
        s1 = s1 - np.mean(s1)
        s1 = s1/np.sqrt(np.var(s1))
        s2 = s1**2
        s2 = s2 - np.mean(s2)
        s2 = s2/np.sqrt(np.var(s2))
        #s1 = np.random.normal(loc=0, scale=1, size=self.num_samples)
        #s2 = np.random.normal(loc=0, scale=1, size=self.num_samples)

        shared_1 = np.array([s1, s2])
        shared_2 = np.copy(shared_1)

        # Private noise data
        private_1 = np.random.normal(-.5, 1, (3, self.num_samples))
        private_2 = np.random.normal(.8, 1.5, (3, self.num_samples))

        z_1 = np.concatenate([shared_1, private_1], axis=0)
        z_2 = np.concatenate([shared_2, private_2], axis=0)

        # Compose all channels
        Az_1 = np.matmul(A_1, z_1)
        Az_2 = np.matmul(A_2, z_2)

        # Add non-linearities
        y_1 = np.zeros_like(Az_1)
        y_2 = np.zeros_like(Az_2)

        y_1[0] = 3 * _sigmoid(Az_1[0]) + 0.1 * Az_1[0]
        y_2[0] = 5 * np.tanh(Az_2[0]) + 0.2 * Az_2[0]

        y_1[1] = 5 * _sigmoid(Az_1[1]) + 0.2 * Az_1[1]
        y_2[1] = 2 * np.tanh(Az_2[1]) + 0.1 * Az_2[1]

        y_1[2] = 0.2 * np.exp(Az_1[2])
        y_2[2] = 0.1 * Az_2[2]**3 + Az_2[2]

        y_1[3] = -4 * _sigmoid(Az_1[3]) - 0.3 * Az_1[3]
        y_2[3] = -5 * np.tanh(Az_2[3]) - 0.4 * Az_2[3]

        y_1[4] = -3 * _sigmoid(Az_1[4]) - 0.2 * Az_1[4]
        y_2[4] = -6 * np.tanh(Az_2[4]) - 0.3 * Az_2[4]

        return tf.convert_to_tensor(y_1, dtype=tf.float32), tf.convert_to_tensor(y_2, dtype=tf.float32), \
            tf.convert_to_tensor(Az_1, dtype=tf.float32), tf.convert_to_tensor(Az_2, dtype=tf.float32), \
            tf.convert_to_tensor(z_1, dtype=tf.float32), tf.convert_to_tensor(z_2, dtype=tf.float32)

    @staticmethod
    def plot_shared_components(z_1, z_2):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].title.set_text(f'View {0}')
        axes[0].scatter(z_1[0], z_1[1])

        axes[1].title.set_text(f'View {1}')
        axes[1].scatter(z_2[0], z_2[1])

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_non_linearities(y_1, y_2, Az_1, Az_2):
        fig, axes = plt.subplots(5, 2, figsize=(10, 15))

        for c in range(5):
            axes[c, 0].title.set_text(f'View {0} Channel {c}')
            axes[c, 0].scatter(Az_1[c], y_1[c], label=r'$\mathrm{g}$')

            axes[c, 1].title.set_text(f'View {1} Channel {c}')
            axes[c, 1].scatter(Az_2[c], y_2[c], label=r'$\mathrm{g}$')

        plt.tight_layout()
        plt.show()
