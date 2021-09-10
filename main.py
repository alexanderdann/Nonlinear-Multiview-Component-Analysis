import tensorflow as tf
from tqdm import tqdm
import visualkeras
import matplotlib.patheffects as PathEffects
from models import BatchPreparation, NonlinearComponentAnalysis, CCA
import matplotlib.pyplot as plt
import time
import os
from TwoChannelModel import *

keys = time.asctime(time.localtime(time.time())).split()

# Please change your folder
path = '/Users/alexander/Documents/Uni/Work/NMCA/Simulation/' + str('-'.join(keys[0:3]))


try:
    os.makedirs(path)
    print(f'Path: {path} exists: {os.path.exists(path)}\n\n')
except:
    pass

rhos = [0.9, 0.75, 0.0]
batch_size = 1024
samples = 1024
z_dim = 2
c_dim = 3
num_views = 2
epochs = 5000
assert z_dim == 2

autoencoder_dims = [(1, None), (256, 'relu'), (1, None)]

# Choose Parabola or Gaussian for relationship between the latent sources
# If transformation = True => Y = g(As) where g is a non-linear function
TCM = TwoChannelModel(
    path=path,
    observations=samples,
    mixing_dim=int(z_dim + c_dim),
    shared_dim=z_dim,
    private_dim=c_dim,
    mode='Parabola',
    transformation=True,
    rhos=rhos)

X, Y, S_x, S_y, created_rhos = TCM.getitems()

def train_neutral_network(epochs, num_views, num_channels, encoder_dims, decoder_dims, samples, plot_path, shared_dim):
    batched_X = BatchPreparation(batch_size=batch_size, samples=samples, data=X)
    batched_Y = BatchPreparation(batch_size=batch_size, samples=samples, data=Y)
    assert tf.shape(batched_X)[2] == tf.shape(batched_Y)[2]
    assert batch_size == tf.shape(batched_X)[2]

    batch_dims = tf.shape(batched_X)[0]
    data_dim = tf.shape(batched_X)[1]

    print('--- Information about the intermediate data ---\n')
    print(f'Amount of Batches: {batch_dims}\nBatch Size: {batch_size}\nData Dimension: {data_dim}\n')

    NCA_Class = NonlinearComponentAnalysis(num_views=num_views,
                                     num_channels=num_channels,
                                     encoder_layers=encoder_dims,
                                     decoder_layers=decoder_dims,
                                     batch_size=batch_size)
    NCA_Model = NCA_Class.NCA

    loss_arr = []
    for batch_idx in tqdm(range(batch_dims), desc='Batch ID'):
        chunkX = tf.transpose(batched_X[batch_idx])
        chunkY = tf.transpose(batched_Y[batch_idx])
        chunkXandY = tf.concat([chunkX, chunkY], 1)

        data_chunk = [chunkXandY[:,i][None] for i in range(2*data_dim)]

        for epoch in tqdm(range(epochs), desc='Epochs'):
            #print(f'######## Batch {batch_idx+1}/{batch_dims} ########')
            #print(f'######## Epoch {epoch+1}/{epochs} ########')
            with tf.GradientTape() as tape:
                tape.watch(data_chunk)
                output_of_encoders, output_of_decoders = NCA_Model(data_chunk)

                c_loss = NCA_Class.loss(output_of_encoders[0][0], output_of_encoders[1][0],
                                      output_of_decoders[0][0], output_of_decoders[1][0],
                                      chunkX, chunkY, batch_size, shared_dim)
                loss_arr.append(c_loss)

            gradients = tape.gradient(c_loss, NCA_Model.trainable_variables)

            NCA_Class.optimizer.apply_gradients(zip(gradients, NCA_Model.trainable_variables))


    # we skip first 20 percent of loss visualisation since it drops very fast and is messing up the plot
    offset = int(len(loss_arr)*0.2)
    plt.plot(np.squeeze([np.linspace(offset, len(loss_arr[offset:])-1, len(loss_arr[offset:]))]), loss_arr[offset:])
    plt.title(r'Loss')
    plt.show()
    eval_data_np, test_sample, S_x, S_y = TCM.eval(batch_size, num_channels, data_dim)
    eval_data_tf = tf.convert_to_tensor(eval_data_np, dtype=tf.float32)
    output_of_encoders, output_of_decoders = NCA_Model([eval_data_tf[i] for i in range(2*data_dim)])

    fig, axes = plt.subplots(num_channels, num_views, figsize=(6, 9))

    for c in range(num_channels):
        for v in range(num_views):
            axes[c, v].title.set_text(f'View {v} Channel {c}')
            axes[c, v].plot(test_sample, output_of_encoders[v][0][:, c].numpy(), label=r'$\mathrm{f}\circledast\mathrm{g}$')
            if v == 0:
                axes[c, v].plot(test_sample, np.squeeze(eval_data_np[:5][c]), label=r'$\mathrm{g}$')
            elif v == 1:
                axes[c, v].plot(test_sample, np.squeeze(eval_data_np[5:][c]), label=r'$\mathrm{g}$')

            axes[c, v].legend()

    plt.tight_layout()
    #full_path = path + '/' + plot_path + f'_{epochs}_Epochs.png'
    #plt.savefig(full_path)
    plt.show()

    plt.scatter(NCA_Class.est_sources[0][0], NCA_Class.est_sources[0][1])
    plt.ylabel(r'$\hat{\mathbf{\varepsilon}}^{(1)}$', fontsize='15')
    plt.xlabel(r'$\hat{\mathbf{\varepsilon}}^{(0)}$', fontsize='15')
    plt.suptitle(r'Estimated Sources $\hat{\mathbf{\varepsilon}}$', fontsize='18')
    plt.tight_layout()
    plt.show()

    plt.scatter(NCA_Class.est_sources[1][0], NCA_Class.est_sources[1][1])
    plt.ylabel(r'$\hat{\mathbf{\omega}}^{(1)}$', fontsize='15')
    plt.xlabel(r'$\hat{\mathbf{\omega}}^{(0)}$', fontsize='15')
    plt.suptitle(r'Estimated Sources $\hat{\mathbf{\omega}}$', fontsize='18')
    plt.tight_layout()
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
    plt.show()

    assert samples == batch_size

    fig, axes = plt.subplots(shared_dim, num_views)

    for s in range(shared_dim):
        for v in range(num_views):
            if v == 0:
                axes[s, v].scatter(S_x[s], NCA_Class.est_sources[v][s], s=4)
                xlab = '$\mathbf{s}_{\mathrm{X}}^{('+ str(s) +')}$'
                ylab = r'$\hat{\mathbf{\varepsilon}}^{('+ str(s) +')}$'
                axes[s, v].set_xlabel(xlab)
                axes[s, v].set_ylabel(ylab)
            elif v == 1:
                axes[s, v].scatter(S_y[s], NCA_Class.est_sources[v][s], s=4)
                xlab = '$\mathbf{s}_{\mathrm{Y}}^{(' + str(s) + ')}$'
                ylab = r'$\hat{\mathbf{\omega}}^{(' + str(s) + ')}$'
                axes[s, v].set_xlabel(xlab)
                axes[s, v].set_ylabel(ylab)

    plt.suptitle('Relationship between the estimated and true sources')
    plt.tight_layout()
    plt.show()

    Cov_SE, dim1, dim2 = PCC_Matrix(tf.constant(S_x, tf.float32), NCA_Class.est_sources[0], samples)

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

    plt.show()

    Cov_SO, dim1, dim2 = PCC_Matrix(tf.constant(S_y, tf.float32), NCA_Class.est_sources[1], samples)

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

    plt.show()



def PCC_Matrix(view1, view2, observations):
    assert tf.shape(view1)[1] == observations
    assert tf.shape(view1)[1] == tf.shape(view2)[1]

    calc_cov = np.zeros([tf.shape(view1)[0], tf.shape(view2)[0]])

    for dim1 in range(tf.shape(view1)[0]):
        for dim2 in range(tf.shape(view2)[0]):
            mu_1 = tf.reduce_mean(view1[dim1])
            mu_2 = tf.reduce_mean(view2[dim2])
            sigma_1 = tf.math.sqrt(tf.reduce_sum([(x - mu_1) ** 2 for x in view1[dim1]]))
            sigma_2 = tf.math.sqrt(tf.reduce_sum([(x - mu_2) ** 2 for x in view2[dim2]]))
            tmp = tf.reduce_sum([(view1[dim1][i]-mu_1)*(view2[dim2][i]-mu_2) for i in range(observations)])/(sigma_1*sigma_2)
            calc_cov[dim1, dim2] = np.abs(tmp.numpy())

    return calc_cov, tf.shape(view1)[0], tf.shape(view2)[0]

for it in range(1):
    path_add = f'Iteration_{it}'
    train_neutral_network(
        epochs=epochs,
        num_views=num_views,
        num_channels=z_dim+c_dim,
        encoder_dims=autoencoder_dims,
        decoder_dims=autoencoder_dims,
        samples=samples,
        plot_path=path_add,
        shared_dim=z_dim)

if __name__ == '__main__':
    print(tf.version)
