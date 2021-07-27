import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
import math
from numpy.random import default_rng

sns.set()
sns.set_style('white')
sns.set_context('paper')

class TwoChannelModel():
    def __init__(self, path, rhos, observations, dim, SM1=0, SM2=0):
        print("----- TwoChannelModel -----\n")

        self.path = path
        np.random.seed(3333)
        num_comp = len(rhos)

        self.A_x_SM = np.array(SM1)
        self.A_y_SM = np.array(SM2)

        self.A_x = np.random.randn(num_comp, dim)
        self.A_y = np.random.randn(num_comp, dim)

        self.mv_samples = []
        for num in range(len(rhos)):
            mean = np.array([0, 0])

            cov = np.array([[1, 0],
                            [0, 1]])

            multivar_sample = np.random.multivariate_normal(mean, cov, size=dim, check_valid='warn', tol=1e-10)
            self.mv_samples.append(multivar_sample.T)

        for i in range(len(rhos)):
            self.A_x[i] = self.mv_samples[i][0]
            self.A_y[i] = self.mv_samples[i][1]


        self.S_x = np.random.randn(observations, num_comp)
        self.S_y = np.random.randn(observations, num_comp)

        self.mv_samples = []
        for num in range(len(rhos)):
            mean = np.array([0, 0])

            cov = np.array([[1, rhos[num]],
                            [rhos[num], 1]])

            multivar_sample = np.random.multivariate_normal(mean, cov, size=observations, check_valid='warn', tol=1e-10)
            self.mv_samples.append(multivar_sample.T)

        for i in range(len(rhos)):
            self.S_x.T[i] = self.mv_samples[i][0]
            self.S_y.T[i] = self.mv_samples[i][1]

        print("Rows of Spatial Maps: " + str(len(self.A_x)))
        print("Columns of Spatial Maps: " + str(len(self.A_x[0])) + "\n")

        print("Number of Analyzed Brain Regions: " + str(len(self.S_x[0])))
        print("Number of Observations: " + str(len(self.S_x)) + "\n")

    def SNR(self, signal, name, noise_var):
        # if len(signal) > 0:
        #    signal = signal.flatten()
        #    noise = noise.flatten()
        signal = signal.T

        N = len(signal[0])
        M = len(signal)
        SNR_per_comp = np.empty([M, 1])

        for m in range(M):
            # signal_mag, noise_mag = [], []
            # print(N)
            sig_var = np.mean([(signal[m][n]) ** 2 for n in range(N)])  # /N
            # sig_var1 = sum([(signal[m][n]) ** 2 for n in range(N)])/N
            # noise_var = np.sum([(noise[n])**2 for n in range(N)])/N
            # print(f'Noise Var: {noise_var}')
            # print(f'Signal Var: {sig_var}')
            # for n in range(N):
            # signal_mag.append(np.linalg.norm(signal[m][n]))
            # signal_mag[n] = signal_mag[n] ** 2
            # noise_mag.append(np.linalg.norm(noise[m][n]))
            # noise_mag[n] = noise_mag[n] ** 2

            # variance_Source = np.sum(signal_mag)
            # Power_Noise = np.sum(noise_mag)
            Ratio = sig_var / noise_var[m]
            # print(f'SNR Ratio: {Ratio}')
            # cor = sig_var/(sig_var+noise_var)
            # print("\n"+str(cor))
            SNR_per_comp[m] = 10 * np.log10(Ratio)

        print(SNR_per_comp)
        np.sort(SNR_per_comp, axis=1)

        print('\nSNR of Signal ' + name + '\n')
        for elem in range(len(SNR_per_comp)):
            print(f'Component {elem}: {np.round(SNR_per_comp[::-1][elem], 3)[0]} dB')

        plt.scatter(np.linspace(0.5, len(SNR_per_comp), len(SNR_per_comp)), SNR_per_comp[::-1])
        plt.show()

    def transform(self, part_mix, mixing1="def", mixing2="def", amp=1., stretch=1., degree=0):
        np.random.seed(3333)

        X = np.dot(self.S_x, self.A_x).T
        Y = np.dot(self.S_y, self.A_y).T

        self.created_rhos = self._PCC(self.S_x, self.S_y)

        print("Generated Signal of Dimensions {0} X {1} \n".format(len(X), len(X[0])))

        if mixing1 != "def":
            if mixing1 == 'poly':
                X, mix1 = self._poly_transformation(X, mixing1, amp, stretch, part_mix, degree)
            else:
                X, mix1 = self._nonlinear_transformation(X, mixing1, amp, stretch, part_mix)
        else:
            print("\nMixing: default\n")
            mix1 = 'Linear'

        if mixing2 != "def":
            if mixing1 == 'poly':
                Y, mix2 = self._poly_transformation(Y, mixing2, amp, stretch, part_mix, degree)
            else:
                Y, mix2 = self._nonlinear_transformation(Y, mixing2, amp, stretch, part_mix)
        else:
            print("\nMixing: default\n")
            mix2 = 'Linear'

        X=X.T
        Y=Y.T

        mix = mix1 + ' and ' + mix2

        plt.rcParams.update({'figure.figsize': (5, 4)})
        # plt.suptitle('Relationship between True Sources $\mathbf{S}_{\mathrm{X}}$ and $\mathbf{S}_{\mathrm{Y}}$', fontweight='bold', fontsize=19)
        title = 'Transformation: ' + mix
        plt.title(title, fontsize='14')
        plt.ylabel('$\mathbf{X}$', fontweight='bold', fontsize='18')
        plt.xlabel('$\mathbf{Y}$', fontweight='bold', fontsize='18')
        legend = plt.scatter(X.T[0], Y.T[0], c='black', marker='.')
        # legend.set_label('rhos=[1, 1, 1]')
        # legend = plt.scatter(self.TC_x, self.test, c='black', marker='.')
        # legend.set_label('rhos=[0.95, 0.95, 0.95]')
        # plt.legend()
        plt.xlim(-10, 10)
        plt.tight_layout()
        full_path = self.path + '/' + 'GENSIG.png'
        plt.savefig(full_path)
        plt.show(block=False)
        plt.close('all')

        #print(f'Covariance of E[X.T * Y]:\n{np.dot(X.T, Y)/len(X)}')

        return X, Y, self.S_x, self.S_y, self.created_rhos

    def _PCC(self, TC_x, TC_y):
        calc_cov = []

        print(f'self.TC_y {TC_y.shape}')

        for i in range(len(TC_y.T)):
            sigma_y = np.sqrt(np.array(sum([y ** 2 for y in TC_y.T[i]])) / len(TC_y))
            sigma_x = np.sqrt(np.array(sum([x ** 2 for x in TC_x.T[i]])) / len(TC_x))
            calc_cov.append(np.dot(TC_x.T[i], TC_y.T[i]) / (len(TC_x) * sigma_y * sigma_x))

        calc_cov = np.sort(calc_cov)
        calc_cov = calc_cov[::-1].copy()

        for cor in range(len(calc_cov)):
            if calc_cov[cor] > 1:
                calc_cov[cor] = 1
            elif calc_cov[cor] < 0:
                calc_cov[cor] = 0

        print(f'That are the computed correlations: {calc_cov}')

        return calc_cov

    def _poly_transformation(self, source, mode, amp, stretch, part_mix, degree):
        modified_source = source

        params = [1, 1.1, 1, -1, -0.3, 0.2, -0.0215, 0.0004]

        if degree == 1:
            func = lambda x: 1 * x + 0

        elif degree == 2:
            func = lambda x: -x * np.sin(0.2 * x)

        elif degree == 3:
            func = lambda x: -np.sin(0.9 * x)

        elif degree == 4:
            func = lambda x: 1.5 * x * np.sin(0.6 * x) ** 3 * (np.cos(0.4 * x))

        elif degree == 5:
            func = lambda x: 1.5 * x * np.sin(1.6 * (x - 3))

        elif degree == 6:
            func = lambda x: 1.5 * x * np.sin(1.5 * x) ** 3 * (np.cos(0.4 * x))

        elif degree == 7:
            func = lambda x: 1.5 * x * np.sin(1.9 * (x - 3)) * np.cos(0.5 * x)

        elif degree == 8:
            func = lambda x: 1.5 * x * np.sin(1.9 * (x - 3)) * np.cos(.75 * x)

        elif degree == 9:
            func = lambda x: 1.5 * x * np.sin(1.9 * (x - 3)) * np.cos(1.1 * x)

        elif degree == 10:
            func = lambda x: 1.5 * x * np.sin(2.3 * (x - 3)) * np.cos(1.1 * x)

        for i in range(len(modified_source)):
            for j in part_mix:
                erg = 0
                for d in range(degree + 1):
                    erg += func(modified_source[i][j - 1])
                modified_source[i][j - 1] = erg

        if degree == 0:
            for i in modified_source:
                for j in i:
                    if j > 0.99 or j < 1.01:
                        print(True)
                    else:
                        print(False)

        print(modified_source)
        return modified_source, 'Polynomial'

    def _nonlinear_transformation(self, source, mode, amp, stretch, part_mix):
        # Possibilities: exp, ELU, softplus, sinh, cubic, ccubic, sin, tanh
        if mode == "exp":
            print("\nExponential transformation was chosen.\n")

            modified_source = amp * source

            for i in range(len(modified_source)):
                for j in part_mix:
                    modified_source[i][j - 1] = amp * np.exp(modified_source[i][j - 1])

            return modified_source, 'Exponential'

        elif mode == 'sinh':
            sinh = lambda x: 0.5 * (np.exp(x) - np.exp(-x))

            print("\nSinh transformation was chosen.\n")

            modified_source = amp * source

            for i in range(len(modified_source)):
                for j in part_mix:
                    modified_source[i][j - 1] = amp * sinh(modified_source[i][j - 1])

            return modified_source, 'Sinh'


        elif mode == 'ELU':
            ELU = lambda x: x if x >= 0 else 0.1 * (np.exp(x) - 1)

            print("\nELU transformation was chosen.\n")

            modified_source = amp * source

            for i in range(len(modified_source)):
                for j in part_mix:
                    modified_source[i][j - 1] = amp * ELU(modified_source[i][j - 1])

            return modified_source, 'ELU'

        elif mode == 'softplus':
            SoftPlus = lambda x: np.log(1 + np.exp(x))

            print("\nSoftplus transformation was chosen.\n")

            modified_source = amp * source

            for i in range(len(modified_source)):
                for j in part_mix:
                    modified_source[i][j - 1] = amp * SoftPlus(modified_source[i][j - 1])

            return modified_source, 'Softplus'

        elif mode == 'spec':
            print("\nSpecial mixing was chosen.\n")
            modified_source = stretch * source

            for i in range(len(modified_source)):
                for j in part_mix:
                    if i == 0:
                        modified_source[i][j - 1] = amp * np.sin(math.radians(modified_source[i][j - 1]))
                    elif i == 1:
                        modified_source[i][j - 1] = amp * np.exp(modified_source[i][j - 1])
                    else:
                        # linear mixing
                        pass

            return modified_source, 'Special'

        elif mode == 'cubic':
            print("\nCubic mixing was chosen.\n")

            modified_source = amp * source

            for i in range(len(modified_source)):
                for j in part_mix:
                    modified_source[i][j - 1] = (modified_source[i][j - 1]) ** 3

            return modified_source, 'Shifted Cubic'


        elif mode == 'ccubic':

            print("\nCustom Cubic mixing was chosen.\n")

            modified_source = source

            for i in range(len(modified_source)):

                for j in part_mix:

                    if modified_source[i][j - 1] >= 0:

                        modified_source[i][j - 1] = 0

                    else:

                        modified_source[i][j - 1] = (modified_source[i][j - 1]) ** 9

            return modified_source, 'Custom Cubic'

        elif mode == "descsin":
            print("\nDescsin mixing was chosen.\n")

            modified_source = stretch * source

            for i in range(len(modified_source)):
                for j in part_mix:
                    modified_source[i][j - 1] = i * np.sin(math.radians(modified_source[i][j - 1]))

            return modified_source, 'Descending Sin'

        elif mode == "sin":
            print("\nSinusoidal mixing was chosen.\n")

            modified_source = stretch * source

            for i in range(len(modified_source)):
                for j in part_mix:
                    modified_source[i][j - 1] = amp * np.sin(math.radians(modified_source[i][j - 1]))

            return modified_source, 'Sinusoidal'

        elif mode == 'reci':
            print("\nReciprocal mixing was chosen.\n")

            modified_source = amp * source

            for i in range(len(modified_source)):
                for j in part_mix:
                    if 1 / modified_source[i][j - 1] >= 30000:
                        pass
                    else:
                        modified_source[i][j - 1] = 1 / modified_source[i][j - 1]

            return modified_source, 'Reciprocal'

        elif mode == 'tanh':
            print("\nTanh mixing was chosen.\n")

            modified_source = amp * source

            for i in range(len(modified_source)):
                for j in part_mix:
                    modified_source[i][j - 1] = amp * np.tanh(modified_source[i][j - 1])

            return modified_source, 'Tanh'

        elif mode == 'sqr':
            print("\nSquareroot mixing was chosen.\n")

            modified_source = stretch * source

            for i in range(len(modified_source)):
                for j in part_mix:
                    if modified_source[i][j - 1] > 0:
                        modified_source[i][j - 1] = np.sqrt(modified_source[i][j - 1])
                    else:
                        modified_source[i][j - 1] = -1 * np.sqrt(-1 * modified_source[i][j - 1])

            return modified_source, 'Normed Square Root'

        else:
            print("\nNO valid nonlinear mixing chosen\n")

