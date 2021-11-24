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

def write_image_summary(writer, epoch, Az_1, Az_2, y_1, y_2, fy_1, fy_2, yhat_1, yhat_2):
    with writer.as_default():
        # Inverse learning plot
        inv_fig, inv_axes = plt.subplots(5, 2, figsize=(10, 15))
        for c in range(5):
            inv_axes[c, 0].title.set_text(f'View {1} Channel {c+1}')
            inv_axes[c, 0].scatter(Az_1[c], tf.transpose(fy_1)[c], label=r'$\mathrm{f}\circledast\mathrm{g}$')

            inv_axes[c, 1].title.set_text(f'View {2} Channel {c+1}')
            inv_axes[c, 1].scatter(Az_2[c], tf.transpose(fy_2)[c], label=r'$\mathrm{f}\circledast\mathrm{g}$')

        plt.tight_layout()
        tf.summary.image("Inverse learning", plot_to_image(inv_fig), step=epoch)
        writer.flush()

        # Reconstruction plot
        rec_fig, rec_axes = plt.subplots(5, 2, figsize=(10, 15))
        for c in range(5):
            rec_axes[c, 0].title.set_text(f'View {1} Channel {c+1}')
            rec_axes[c, 0].scatter(y_1[c], tf.transpose(yhat_1)[c])

            rec_axes[c, 1].title.set_text(f'View {2} Channel {c+1}')
            rec_axes[c, 1].scatter(y_2[c], tf.transpose(yhat_2)[c])

        plt.tight_layout()
        tf.summary.image("Reconstruction", plot_to_image(rec_fig), step=epoch)
        writer.flush()

def write_PCC_summary(writer, epoch, z_1, z_2, epsilon, omega, samples):
    with writer.as_default():
        fig, axes = plt.subplots(1, 2)
        z_s = (z_1, z_2)
        sources = (epsilon, omega)

        for idx in range(2):
            Cov_SE, dim1, dim2 = PCC_Matrix(tf.constant(z_s[idx], tf.float32), sources[idx], samples)

            legend_1 = axes[idx].imshow(Cov_SE, cmap='Oranges')
            clrbr = fig.colorbar(legend_1, orientation="horizontal", pad=0.15, ax=axes[idx])
            for t in clrbr.ax.get_xticklabels():
                t.set_fontsize(10)
            legend_1.set_clim(0, 1)
            clrbr.set_label(r'Correlation', fontsize=15)
            axes[idx].set_xticks(np.arange(1, dim2, 1))
            axes[idx].set_yticks(np.arange(1, dim1, 1))
            axes[idx].tick_params(
                axis='x',
                which='both',
                bottom=False,
                top=False)

            if idx == 0:
                axes[idx].set_xlabel(r'$\hat{\mathbf{\varepsilon}}$', fontsize=18)
                axes[idx].set_ylabel(r'$\mathbf{z}_{\mathrm{1}}$', fontsize=18)
            else:
                axes[idx].set_xlabel(r'$\hat{\mathbf{\omega}}$', fontsize=18)
                axes[idx].set_ylabel(r'$\mathbf{z}_{\mathrm{2}}$', fontsize=18)

            for i in range(len(Cov_SE[0])):
                for j in range(len(Cov_SE)):
                    c = np.around(Cov_SE[j, i], 2)
                    txt = axes[idx].text(i, j, str(c), va='center', ha='center', color='black', size='x-large')
                    #txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

        plt.tight_layout()
        tf.summary.image("PCC Plot", plot_to_image(fig), step=epoch)
        writer.flush()

def write_scalar_summary(writer, epoch, list_of_tuples):
    with writer.as_default():
        for tup in list_of_tuples:
            tf.summary.scalar(tup[1], tup[0], step=epoch)
    writer.flush()


def write_gradients_summary_sum(writer, epoch, gradients, trainable_variables):
    '''
    :param writer: used for tensorboard
    :param epoch: current epoch of training
    :param gradients: containing all gradients of the model
    :param trainable_variables: containing all information regarding the corresponding gradient

    Filtering the gradients to divide them into different groups
    Groups are:
       - Weights/Kernels only for each Encoder and Decoder
       - Bias only for each Encoder and Decoder
       - Encoder gradients (Combining weights/kernels and bias) for each view
       - Decoder gradients (Combining weights/kernels and bias) for each view

    Note: This functions uses SUMMATION of the gradients as final output
    '''

    view_0_encoder_kernel, view_1_encoder_kernel = list(), list()
    view_0_encoder_bias, view_1_encoder_bias = list(), list()

    view_0_decoder_kernel, view_1_decoder_kernel = list(), list()
    view_0_decoder_bias, view_1_decoder_bias = list(), list()

    for gradient in gradients:
        for gradient_idx, gradient_unit in enumerate(gradient):

            if 'View_0_Encoder' in trainable_variables[gradient_idx].name:
                if 'kernel' in trainable_variables[gradient_idx].name:
                    view_0_encoder_kernel.append(gradient_unit)
                else:
                    view_0_encoder_bias.append(gradient_unit)

            elif 'View_1_Encoder' in trainable_variables[gradient_idx].name:
                if 'kernel' in trainable_variables[gradient_idx].name:
                    view_1_encoder_kernel.append(gradient_unit)
                else:
                    view_1_encoder_bias.append(gradient_unit)

            elif 'View_0_Decoder' in trainable_variables[gradient_idx].name:
                if 'kernel' in trainable_variables[gradient_idx].name:
                    view_0_decoder_kernel.append(gradient_unit)
                else:
                    view_0_decoder_bias.append(gradient_unit)

            elif 'View_1_Decoder' in trainable_variables[gradient_idx].name:
                if 'kernel' in trainable_variables[gradient_idx].name:
                    view_1_decoder_kernel.append(gradient_unit)
                else:
                    view_1_decoder_bias.append(gradient_unit)

    grad_encoder_0_kernel = np.sum([np.linalg.norm(grad) for grad in view_0_encoder_kernel])
    grad_encoder_1_kernel = np.sum([np.linalg.norm(grad) for grad in view_1_encoder_kernel])

    grad_encoder_0_bias = np.sum([np.linalg.norm(grad) for grad in view_0_encoder_bias])
    grad_encoder_1_bias = np.sum([np.linalg.norm(grad) for grad in view_1_encoder_bias])

    grad_decoder_0_kernel = np.sum([np.linalg.norm(grad) for grad in view_0_decoder_kernel])
    grad_decoder_1_kernel = np.sum([np.linalg.norm(grad) for grad in view_1_decoder_kernel])

    grad_decoder_0_bias = np.sum([np.linalg.norm(grad) for grad in view_0_decoder_bias])
    grad_decoder_1_bias = np.sum([np.linalg.norm(grad) for grad in view_1_decoder_bias])

    units = [('Gradient Units SUM/Encoder View 0 Kernel', grad_encoder_0_kernel),
               ('Gradient Units SUM/Encoder View 0 Bias', grad_encoder_0_bias),
               ('Gradient Units SUM/Encoder View 1 Kernel', grad_encoder_1_kernel),
               ('Gradient Units SUM/Encoder View 1 Bias', grad_encoder_1_bias),
               ('Gradient Units SUM/Decoder View 0 Kernel', grad_decoder_0_kernel),
               ('Gradient Units SUM/Decoder View 0 Bias', grad_decoder_0_bias),
               ('Gradient Units SUM/Decoder View 1 Kernel', grad_decoder_1_kernel),
               ('Gradient Units SUM/Decoder View 1 Bias', grad_decoder_1_bias)
               ]

    # Concatenating two lists of norms and taking the sum
    grad_encoder_0 = np.sum([np.linalg.norm(grad) for grad in view_0_encoder_kernel] +
                            [np.linalg.norm(grad) for grad in view_0_encoder_bias])

    grad_encoder_1 = np.sum([np.linalg.norm(grad) for grad in view_1_encoder_kernel] +
                            [np.linalg.norm(grad) for grad in view_1_encoder_bias])

    encoders = [('Gradient Encoders SUM/View 0', grad_encoder_0),
                ('Gradient Encoders SUM/View 1', grad_encoder_1)]

    grad_decoder_0 = np.sum([np.linalg.norm(grad) for grad in view_0_decoder_kernel] +
                            [np.linalg.norm(grad) for grad in view_0_decoder_bias])

    grad_decoder_1 = np.sum([np.linalg.norm(grad) for grad in view_1_decoder_kernel] +
                            [np.linalg.norm(grad) for grad in view_1_decoder_bias])

    decoders = [('Gradient Decoders SUM/View 0', grad_decoder_0),
                ('Gradient Decoders SUM/View 1', grad_decoder_1)]

    with writer.as_default():
        for path, variable in units:
            tf.summary.scalar(path, variable, step=epoch)

        for path, variable in encoders:
            tf.summary.scalar(path, variable, step=epoch)

        for path, variable in decoders:
            tf.summary.scalar(path, variable, step=epoch)

def write_gradients_summary_mean(writer, epoch, gradients, trainable_variables):
    '''
    :param writer: used for tensorboard
    :param epoch: current epoch of training
    :param gradients: containing all gradients of the model
    :param trainable_variables: containing all information regarding the corresponding gradient

    Filtering the gradients to divide them into different groups
    Groups are:
       - Weights/Kernels only for each Encoder and Decoder
       - Bias only for each Encoder and Decoder
       - Encoder gradients (Combining weights/kernels and bias) for each view
       - Decoder gradients (Combining weights/kernels and bias) for each view

    Note: This functions uses MEAN of the gradients as final output
    '''

    view_0_encoder_kernel, view_1_encoder_kernel = list(), list()
    view_0_encoder_bias, view_1_encoder_bias = list(), list()

    view_0_decoder_kernel, view_1_decoder_kernel = list(), list()
    view_0_decoder_bias, view_1_decoder_bias = list(), list()

    for gradient in gradients:
        for gradient_idx, gradient_unit in enumerate(gradient):

            if 'View_0_Encoder' in trainable_variables[gradient_idx].name:
                if 'kernel' in trainable_variables[gradient_idx].name:
                    view_0_encoder_kernel.append(gradient_unit)
                else:
                    view_0_encoder_bias.append(gradient_unit)

            elif 'View_1_Encoder' in trainable_variables[gradient_idx].name:
                if 'kernel' in trainable_variables[gradient_idx].name:
                    view_1_encoder_kernel.append(gradient_unit)
                else:
                    view_1_encoder_bias.append(gradient_unit)

            elif 'View_0_Decoder' in trainable_variables[gradient_idx].name:
                if 'kernel' in trainable_variables[gradient_idx].name:
                    view_0_decoder_kernel.append(gradient_unit)
                else:
                    view_0_decoder_bias.append(gradient_unit)

            elif 'View_1_Decoder' in trainable_variables[gradient_idx].name:
                if 'kernel' in trainable_variables[gradient_idx].name:
                    view_1_decoder_kernel.append(gradient_unit)
                else:
                    view_1_decoder_bias.append(gradient_unit)

    grad_encoder_0_kernel = np.mean([np.linalg.norm(grad) for grad in view_0_encoder_kernel])
    grad_encoder_1_kernel = np.mean([np.linalg.norm(grad) for grad in view_1_encoder_kernel])

    grad_encoder_0_bias = np.mean([np.linalg.norm(grad) for grad in view_0_encoder_bias])
    grad_encoder_1_bias = np.mean([np.linalg.norm(grad) for grad in view_1_encoder_bias])

    grad_decoder_0_kernel = np.mean([np.linalg.norm(grad) for grad in view_0_decoder_kernel])
    grad_decoder_1_kernel = np.mean([np.linalg.norm(grad) for grad in view_1_decoder_kernel])

    grad_decoder_0_bias = np.mean([np.linalg.norm(grad) for grad in view_0_decoder_bias])
    grad_decoder_1_bias = np.mean([np.linalg.norm(grad) for grad in view_1_decoder_bias])

    units = [('Gradient Units MEAN/Encoder View 0 Kernel', grad_encoder_0_kernel),
               ('Gradient Units MEAN/Encoder View 0 Bias', grad_encoder_0_bias),
               ('Gradient Units MEAN/Encoder View 1 Kernel', grad_encoder_1_kernel),
               ('Gradient Units MEAN/Encoder View 1 Bias', grad_encoder_1_bias),
               ('Gradient Units MEAN/Decoder View 0 Kernel', grad_decoder_0_kernel),
               ('Gradient Units MEAN/Decoder View 0 Bias', grad_decoder_0_bias),
               ('Gradient Units MEAN/Decoder View 1 Kernel', grad_decoder_1_kernel),
               ('Gradient Units MEAN/Decoder View 1 Bias', grad_decoder_1_bias)
               ]

    # Concatenating two lists of norms and taking the sum
    grad_encoder_0 = np.mean([np.linalg.norm(grad) for grad in view_0_encoder_kernel] +
                            [np.linalg.norm(grad) for grad in view_0_encoder_bias])

    grad_encoder_1 = np.mean([np.linalg.norm(grad) for grad in view_1_encoder_kernel] +
                            [np.linalg.norm(grad) for grad in view_1_encoder_bias])

    encoders = [('Gradient Encoders MEAN/View 0', grad_encoder_0),
                ('Gradient Encoders MEAN/View 1', grad_encoder_1)]

    grad_decoder_0 = np.mean([np.linalg.norm(grad) for grad in view_0_decoder_kernel] +
                            [np.linalg.norm(grad) for grad in view_0_decoder_bias])

    grad_decoder_1 = np.mean([np.linalg.norm(grad) for grad in view_1_decoder_kernel] +
                            [np.linalg.norm(grad) for grad in view_1_decoder_bias])

    decoders = [('Gradient Decoders MEAN/View 0', grad_decoder_0),
                ('Gradient Decoders MEAN/View 1', grad_decoder_1)]

    with writer.as_default():
        for path, variable in units:
            tf.summary.scalar(path, variable, step=epoch)

        for path, variable in encoders:
            tf.summary.scalar(path, variable, step=epoch)

        for path, variable in decoders:
            tf.summary.scalar(path, variable, step=epoch)