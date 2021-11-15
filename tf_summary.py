from correlation_analysis import CCA, PCC_Matrix
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import io


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def write_poly(writer, epoch, y_1, yhat_1):
    with writer.as_default():
        print(tf.shape(y_1), tf.shape(yhat_1))
        inv_fig, inv_axes = plt.subplots(int(tf.shape(y_1)[0]), 1, figsize=(10, 15))
        for i in range(tf.shape(y_1)[0]):
            inv_axes[i].scatter(y_1[i].numpy(), yhat_1[i].numpy(), s=3, label=f'default')
            for deg in range(1, 6, 2):
                coeff = np.polynomial.polynomial.polyfit(y_1[i].numpy(), yhat_1[i].numpy(), deg)
                inv_axes[i].scatter(y_1[i].numpy(), np.polyval(coeff, y_1[i].numpy()), s=3, label=f'degree {deg}')
            inv_axes[i].set_ylim([yhat_1[i].numpy().min()/3, 3*yhat_1[i].numpy().max()])
            inv_axes[i].legend()

        tf.summary.image("Poly", plot_to_image(inv_fig), step=epoch)
        writer.flush()

def write_image_summary(writer, epoch, Az_1, Az_2, y_1, y_2, fy_1, fy_2):
    with writer.as_default():
        # Inverse learning plot
        inv_fig, inv_axes = plt.subplots(5, 2, figsize=(10, 15))
        for c in range(5):
            inv_axes[c, 0].title.set_text(f'View {0} Channel {c}')
            #inv_axes[c, 0].scatter(Az_1[c], y_1[c], label=r'$\mathrm{g}$')
            inv_axes[c, 0].scatter(Az_1[c], tf.transpose(fy_1)[c], label=r'$\mathrm{f}\circledast\mathrm{g}$')

            inv_axes[c, 1].title.set_text(f'View {1} Channel {c}')
            #inv_axes[c, 1].scatter(Az_2[c], y_2[c], label=r'$\mathrm{g}$')
            inv_axes[c, 1].scatter(Az_2[c], tf.transpose(fy_2)[c], label=r'$\mathrm{f}\circledast\mathrm{g}$')

        plt.legend()
        plt.tight_layout()
        tf.summary.image("Inverse learning", plot_to_image(inv_fig), step=epoch)
        writer.flush()

def write_PCC_summary(writer, epoch, z_1, z_2, epsilon, omega, samples):
    with writer.as_default():
        fig, axes = plt.subplots(1, 2)

        Cov_SE, dim1, dim2 = PCC_Matrix(tf.constant(z_1, tf.float32), epsilon, samples)

        legend_1 = axes[0].imshow(Cov_SE, cmap='Oranges')
        clrbr = fig.colorbar(legend_1, orientation="horizontal", pad=0.15, ax=axes[0])
        for t in clrbr.ax.get_xticklabels():
            t.set_fontsize(10)
        legend_1.set_clim(0, 1)
        clrbr.set_label(r'Correlation', fontsize=15)
        axes[0].set_xlabel(r'$\hat{\mathbf{\varepsilon}}$', fontsize=18)
        axes[0].set_ylabel(r'$\mathbf{z}_{\mathrm{1}}$', fontsize=18)
        axes[0].set_xticks(np.arange(0, dim2, 1))
        axes[0].set_yticks(np.arange(0, dim1, 1))
        axes[0].tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False)

        for i in range(len(Cov_SE[0])):
            for j in range(len(Cov_SE)):
                c = np.around(Cov_SE[j, i], 2)
                txt = axes[0].text(i, j, str(c), va='center', ha='center', color='black', size='x-large')
                # txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

        Cov_SO, dim1, dim2 = PCC_Matrix(tf.constant(z_2, tf.float32), omega, samples)

        legend = axes[1].imshow(Cov_SO, cmap='Oranges')
        clrbr = plt.colorbar(legend, orientation="horizontal", pad=0.15, ax=axes[1])
        for t in clrbr.ax.get_xticklabels():
           t.set_fontsize(10)
        legend.set_clim(0, 1)
        clrbr.set_label(r'Correlation', fontsize=15)
        axes[1].set_xlabel(r'$\hat{\mathbf{\omega}}$', fontsize=18)
        axes[1].set_ylabel(r'$\mathbf{z}_{\mathrm{2}}$', fontsize=18)
        axes[1].set_xticks(np.arange(0, dim2, 1))
        axes[1].set_yticks(np.arange(0, dim1, 1))
        axes[1].tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False)

        for i in range(len(Cov_SO[0])):
            for j in range(len(Cov_SO)):
                c = np.around(Cov_SO[j, i], 2)
                txt = axes[1].text(i, j, str(c), va='center', ha='center', color='black', size='x-large')
                # txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

        plt.tight_layout()
        tf.summary.image("PCC Plot", plot_to_image(fig), step=epoch)
        writer.flush()

def write_metric_summary(writer, epoch, loss, cca_loss, rec_loss, ccor, dist, sim, sim_v1, sim_v2):
    with writer.as_default():
        tf.summary.scalar("Loss/Total", loss, step=epoch)
        tf.summary.scalar("Loss/CCA", cca_loss, step=epoch)
        tf.summary.scalar("Loss/reconstruction", rec_loss, step=epoch)
        tf.summary.scalar("Canonical correlation/0", ccor[0], step=epoch)
        tf.summary.scalar("Canonical correlation/1", ccor[1], step=epoch)
        tf.summary.scalar("Canonical correlation/2", ccor[2], step=epoch)
        tf.summary.scalar("Canonical correlation/3", ccor[3], step=epoch)
        tf.summary.scalar("Canonical correlation/4", ccor[4], step=epoch)
        tf.summary.scalar("Performance Measures/Distance measure", dist, step=epoch)
        tf.summary.scalar("Performance Measures/Similarity measure", sim, step=epoch)
        tf.summary.scalar("Performance Measures/Similarity measure 1st view", sim_v1, step=epoch)
        tf.summary.scalar("Performance Measures/Similarity measure 2nd view", sim_v2, step=epoch)
        writer.flush()

def write_gradients_summary(writer, gradients, trainable_variables):
    # Filtering the gradients to divide them into different groups
    # Groups:   Encoder View 1 & View 2
    #           Decoder View 1 & View 2
    #           Encoder & Decoder View 1 Weights only
    #           Encoder & Decoder View 2 Weights only
    #           Encoder & Decoder View 1 Bias only
    #           Encoder & Decoder View 2 Bias only
    #           Encoder & Decoder View 1 Weights only
    #           Encoder & Decoder View 2 Weights only
    with writer.as_default():
        pass
