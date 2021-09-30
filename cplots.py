import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from metrics import PCC_Matrix
import hickle as hkl
import os
import time

def global_dist(plot_path, labels):
    plt.title('Comparison: Dist Measure over time')
    for lbl in labels:
        data = hkl.load(f'{plot_path}/lambda {lbl}/params.p')
        plt.plot(np.arange(0, len(data[3])), data[3], label=f'Lambda {lbl}')

    plt.legend(bbox_to_anchor=(1, 1.02), ncol=1, fancybox=True, shadow=True)
    plt.xlabel(r'Epochs')
    plt.ylabel(r'Value')
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(f'{plot_path}/Global Dist.png')
    plt.show()

def cplots(dir_path):
    NCA_Model = tf.keras.models.load_model(f'{dir_path}/Model')

    loss_arr, loss_arr_1, loss_arr_2, dist_arr,\
    batch_size, num_channels, num_views, shared_dim, data_dim,\
    epochs, samples, NCA_Class, X_e, Y_e, S_x_e, S_y_e, AS_x_e, AS_y_e = hkl.load(f'{dir_path}/params.p')

    chunkXandY = tf.concat([tf.transpose(X_e), tf.transpose(Y_e)], 1)

    data_chunk = [chunkXandY[:, i][None] for i in range(2 * data_dim)]

    output_of_encoders, output_of_decoders = NCA_Model(data_chunk)

    test_sample = np.linspace(-10, 10, batch_size)
    # we skip first 20 percent of loss visualisation since it drops very fast and is messing up the plot
    offset = int(len(loss_arr)*0.2)
    plt.plot(np.squeeze([np.linspace(offset, len(loss_arr[offset:])-1, len(loss_arr[offset:]))]), loss_arr[offset:],
             label='Complete Loss')
    plt.plot(np.squeeze([np.linspace(offset, len(loss_arr[offset:]) - 1, len(loss_arr[offset:]))]), loss_arr_1[offset:],
             label='Loss for Unconstrained Part')
    plt.plot(np.squeeze([np.linspace(offset, len(loss_arr[offset:]) - 1, len(loss_arr[offset:]))]), loss_arr_2[offset:],
             label='Loss for Regularization Part')
    plt.legend()
    plt.title(r'Loss')
    plt.savefig(f'{dir_path}/Loss.png')
    plt.show()

    #eval_data_np, test_sample, S_x, S_y = TCM.eval(batch_size, num_channels, data_dim)
    #eval_data_tf = tf.convert_to_tensor(eval_data_np, dtype=tf.float32)
    #output_of_encoders, output_of_decoders = NCA_Model([eval_data_tf[i] for i in range(2*data_dim)])

    fig, axes = plt.subplots(num_channels, num_views, figsize=(6, 9))

    for c in range(num_channels):
        for v in range(num_views):
            axes[c, v].title.set_text(f'View {v} Channel {c}')

            if v == 0:
                axes[c, v].scatter(AS_x_e[c], output_of_encoders[v][0][:, c].numpy(),
                                   label=r'$\mathrm{f}\circledast\mathrm{g}$', s=2)
            #    axes[c, v].plot(test_sample, np.squeeze(eval_data_np[:5][c]), label=r'$\mathrm{g}$')
            elif v == 1:
                axes[c, v].scatter(AS_y_e[c], output_of_encoders[v][0][:, c].numpy(),
                                   label=r'$\mathrm{f}\circledast\mathrm{g}$', s=2)
            #    axes[c, v].plot(test_sample, np.squeeze(eval_data_np[5:][c]), label=r'$\mathrm{g}$')

            axes[c, v].legend()

    plt.tight_layout()
    plt.savefig(f'{dir_path}/Inverse Func.png')
    plt.show()

    plt.scatter(NCA_Class.est_sources[0][0], NCA_Class.est_sources[0][1])
    plt.ylabel(r'$\hat{\mathbf{\varepsilon}}^{(1)}$', fontsize='15')
    plt.xlabel(r'$\hat{\mathbf{\varepsilon}}^{(0)}$', fontsize='15')
    plt.suptitle(r'Estimated Sources $\hat{\mathbf{\varepsilon}}$', fontsize='18')
    plt.tight_layout()
    plt.savefig(f'{dir_path}/Estimated Epsilon.png')
    plt.show()

    plt.scatter(NCA_Class.est_sources[1][0], NCA_Class.est_sources[1][1])
    plt.ylabel(r'$\hat{\mathbf{\omega}}^{(1)}$', fontsize='15')
    plt.xlabel(r'$\hat{\mathbf{\omega}}^{(0)}$', fontsize='15')
    plt.suptitle(r'Estimated Sources $\hat{\mathbf{\omega}}$', fontsize='18')
    plt.tight_layout()
    plt.savefig(f'{dir_path}/Estimated Omega.png')
    plt.show()

    x_axis = np.linspace(1, epochs, epochs)
    corrs = len(NCA_Class.can_corr[0])
    labels = [r'Correlation $\rho^{('+ str(i) +')}$' for i in range(corrs)]
    plt.plot(x_axis, NCA_Class.can_corr, label=labels)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.30), ncol=3, fancybox=True, shadow=True)
    plt.ylabel(r'Canonical Correlation value')
    plt.xlabel(r'Epochs')
    plt.yticks(np.arange(0.0, 1.0, 0.2))
    plt.xticks([i for i in range(1, epochs+1) if (i % (0.5*epochs) == 0) or (i==1) or (i==epochs)])
    plt.tight_layout()
    plt.savefig(f'{dir_path}/Correlations.png')
    plt.show()


    plt.plot(np.arange(0, len(dist_arr)), dist_arr)
    plt.title('Dist Measure over time')
    plt.xlabel(r'Epochs')
    plt.ylabel(r'Value')
    plt.ylim([0, 1])
    plt.savefig(f'{dir_path}/Dist.png')
    plt.show()

    assert samples == batch_size

    fig, axes = plt.subplots(shared_dim, num_views)

    for s in range(shared_dim):
        for v in range(num_views):
            if v == 0:
                axes[s, v].scatter(S_x_e[s], NCA_Class.est_sources[v][s], s=4)
                xlab = r'$\mathbf{s}_{\mathrm{X}}^{('+ str(s) + ')}$'
                ylab = r'$\hat{\mathbf{\varepsilon}}^{('+ str(s) + ')}$'
                axes[s, v].set_xlabel(xlab)
                axes[s, v].set_ylabel(ylab)
            elif v == 1:
                axes[s, v].scatter(S_y_e[s], NCA_Class.est_sources[v][s], s=4)
                xlab = r'$\mathbf{s}_{\mathrm{Y}}^{(' + str(s) + ')}$'
                ylab = r'$\hat{\mathbf{\omega}}^{(' + str(s) + ')}$'
                axes[s, v].set_xlabel(xlab)
                axes[s, v].set_ylabel(ylab)

    plt.suptitle('Relationship between the estimated and true sources')
    plt.tight_layout()
    plt.savefig(f'{dir_path}/Scatter Sources.png')
    plt.show()

    Cov_SE, dim1, dim2 = PCC_Matrix(tf.constant(S_x_e, tf.float32), NCA_Class.est_sources[0], samples)

    fig, ax = plt.subplots(figsize=(4, 6))
    legend = ax.imshow(Cov_SE, cmap='Oranges')
    clrbr = plt.colorbar(legend, orientation="horizontal", pad=0.15)
    for t in clrbr.ax.get_xticklabels():
        t.set_fontsize(10)
    legend.set_clim(0, 1)
    clrbr.set_label(r'Correlation', fontsize=   15)
    plt.xlabel(r'$\hat{\mathbf{\varepsilon}}$', fontsize=18)
    plt.ylabel(r'$\mathbf{S}_{\mathrm{X}}$', fontsize=18)
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
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

    plt.savefig(f'{dir_path}/Correlations X.png')
    plt.show()

    Cov_SO, dim1, dim2 = PCC_Matrix(tf.constant(S_y_e, tf.float32), NCA_Class.est_sources[1], samples)

    fig, ax = plt.subplots(figsize=(4, 6))
    legend = ax.imshow(Cov_SO, cmap='Oranges')
    clrbr = plt.colorbar(legend, orientation="horizontal", pad=0.15)
    for t in clrbr.ax.get_xticklabels():
        t.set_fontsize(10)
    legend.set_clim(0, 1)
    clrbr.set_label(r'Correlation', fontsize=15)
    plt.xlabel(r'$\hat{\mathbf{\omega}}$', fontsize=18)
    plt.ylabel(r'$\mathbf{S}_{\mathrm{Y}}$', fontsize=18)
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
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

    plt.savefig(f'{dir_path}/Correlations Y.png')
    plt.show()



keys = time.asctime(time.localtime(time.time())).split()
path = os.getcwd() + '/Final/' + str('-'.join(keys[0:3]))

#hidden_dims = [256, 128, 64, 32]
#lambda_regs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

hidden_dims = [256]
lambda_regs = [0.005]

for it in range(0, 5):
    for hidd_dim in hidden_dims:
        plot_path = f'{path}/Iteration {it}/Hidden Dimension {hidd_dim}'
        global_dist(plot_path, lambda_regs)

        for lamb_val in lambda_regs:
            full_path = f'{path}/Iteration {it}/Hidden Dimension {hidd_dim}/lambda {lamb_val}'
            cplots(full_path)
