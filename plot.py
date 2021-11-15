import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from correlation_analysis import PCC_Matrix


def plot_eval(z_1, z_2, Az_1, Az_2, y_1, y_2, fy_1, fy_2, yhat_1, yhat_2, epsilon, omega):

    print("--------------------------")
    print("Reconstruction performance")
    print("--------------------------")

    fig, axes = plt.subplots(5, 2, figsize=(10, 15))

    for c in range(5):
        axes[c, 0].title.set_text(f'View {0} Channel {c}')
        axes[c, 0].scatter(y_1[c], yhat_1[c])

        axes[c, 1].title.set_text(f'View {1} Channel {c}')
        axes[c, 1].scatter(y_2[c], yhat_2[c])

    plt.tight_layout()
    plt.show()

    print("--------------------------")
    print("Inverse learning")
    print("--------------------------")

    fig, axes = plt.subplots(5, 2, figsize=(10, 15))
    for c in range(5):
        axes[c, 0].title.set_text(f'View {0} Channel {c}')
        axes[c, 0].scatter(Az_1[c], fy_1[c], label=r'$\mathrm{f}\circledast\mathrm{g}$')

        axes[c, 1].title.set_text(f'View {1} Channel {c}')
        axes[c, 1].scatter(Az_2[c], fy_2[c], label=r'$\mathrm{f}\circledast\mathrm{g}$')

    plt.tight_layout()
    plt.show()

    print("--------------------------")
    print("Estimated sources")
    print("--------------------------")

    plt.scatter(epsilon[0], epsilon[1])
    plt.ylabel(r'$\hat{\mathbf{\varepsilon}}^{(1)}$', fontsize='15')
    plt.xlabel(r'$\hat{\mathbf{\varepsilon}}^{(0)}$', fontsize='15')
    plt.suptitle(r'Estimated Sources $\hat{\mathbf{\varepsilon}}$', fontsize='18')
    plt.tight_layout()
    plt.show()

    plt.scatter(omega[0], omega[1])
    plt.ylabel(r'$\hat{\mathbf{\omega}}^{(1)}$', fontsize='15')
    plt.xlabel(r'$\hat{\mathbf{\omega}}^{(0)}$', fontsize='15')
    plt.suptitle(r'Estimated Sources $\hat{\mathbf{\omega}}$', fontsize='18')
    plt.tight_layout()
    plt.show()

    print("--------------------------")
    print("Estimated and true sources")
    print("--------------------------")

    fig, axes = plt.subplots(2, 2)
    for s in range(2):
        axes[s, 0].scatter(z_1[s], epsilon[s], s=4)
        xlab = '$\mathbf{z}_{\mathrm{1}}^{(' + str(s) + ')}$'
        ylab = r'$\hat{\mathbf{\varepsilon}}^{(' + str(s) + ')}$'
        axes[s, 0].set_xlabel(xlab)
        axes[s, 0].set_ylabel(ylab)

        axes[s, 1].scatter(z_2[s], omega[s], s=4)
        xlab = '$\mathbf{z}_{\mathrm{2}}^{(' + str(s) + ')}$'
        ylab = r'$\hat{\mathbf{\omega}}^{(' + str(s) + ')}$'
        axes[s, 1].set_xlabel(xlab)
        axes[s, 1].set_ylabel(ylab)

    plt.suptitle('Relationship between the estimated and true sources')
    plt.tight_layout()
    plt.show()

    print("--------------------------")
    print("PCC matrices")
    print("--------------------------")

    Cov_SE, dim1, dim2 = PCC_Matrix(tf.constant(z_1, tf.float32), epsilon, 1000)

    fig, ax = plt.subplots(figsize=(4, 6))
    legend = ax.imshow(Cov_SE, cmap='Oranges')
    clrbr = plt.colorbar(legend, orientation="horizontal", pad=0.15)
    for t in clrbr.ax.get_xticklabels():
        t.set_fontsize(10)
    legend.set_clim(0, 1)
    clrbr.set_label(r'Correlation', fontsize=15)
    plt.xlabel(r'$\hat{\mathbf{\varepsilon}}$', fontsize=18)
    plt.ylabel(r'$\mathbf{z}_{\mathrm{1}}$', fontsize=18)
    plt.xticks(np.arange(0, dim2, 1), labels=np.arange(0, dim2, 1), fontsize=12)
    plt.yticks(np.arange(0, dim1, 1), np.arange(0, dim1, 1), fontsize=12)
    plt.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False)

    for i in range(len(Cov_SE[0])):
        for j in range(len(Cov_SE)):
            c = np.around(Cov_SE[j, i], 2)
            txt = ax.text(i, j, str(c), va='center', ha='center', color='black', size='x-large')
            #txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

    plt.show()

    Cov_SO, dim1, dim2 = PCC_Matrix(tf.constant(z_2, tf.float32), omega, 1000)

    fig, ax = plt.subplots(figsize=(4, 6))
    legend = ax.imshow(Cov_SO, cmap='Oranges')
    clrbr = plt.colorbar(legend, orientation="horizontal", pad=0.15)
    for t in clrbr.ax.get_xticklabels():
        t.set_fontsize(10)
    legend.set_clim(0, 1)
    clrbr.set_label(r'Correlation', fontsize=15)
    plt.xlabel(r'$\hat{\mathbf{\omega}}$', fontsize=18)
    plt.ylabel(r'$\mathbf{z}_{\mathrm{2}}$', fontsize=18)
    plt.xticks(np.arange(0, dim2, 1), labels=np.arange(0, dim2, 1), fontsize=12)
    plt.yticks(np.arange(0, dim1, 1), np.arange(0, dim1, 1), fontsize=12)
    plt.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False)

    for i in range(len(Cov_SO[0])):
        for j in range(len(Cov_SO)):
            c = np.around(Cov_SO[j, i], 2)
            txt = ax.text(i, j, str(c), va='center', ha='center', color='black', size='x-large')
            #txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

    plt.show()
