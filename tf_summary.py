import tensorflow as tf
import matplotlib.pyplot as plt
import io


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def write_image_summary(writer, epoch, Az_1, Az_2, y_1, y_2, fy_1, fy_2):
    with writer.as_default():
        # Inverse learning plot
        inv_fig, inv_axes = plt.subplots(5, 2, figsize=(10, 15))
        for c in range(5):
            inv_axes[c, 0].title.set_text(f'View {0} Channel {c}')
            inv_axes[c, 0].scatter(Az_1[c], y_1[c], label=r'$\mathrm{g}$')
            inv_axes[c, 0].scatter(Az_1[c], tf.transpose(fy_1)[c], label=r'$\mathrm{f}\circledast\mathrm{g}$')

            inv_axes[c, 1].title.set_text(f'View {1} Channel {c}')
            inv_axes[c, 1].scatter(Az_2[c], y_2[c], label=r'$\mathrm{g}$')
            inv_axes[c, 1].scatter(Az_2[c], tf.transpose(fy_2)[c], label=r'$\mathrm{f}\circledast\mathrm{g}$')
        tf.summary.image("Inverse learning", plot_to_image(inv_fig), step=epoch)
        writer.flush()


def write_metric_summary(writer, epoch, loss, cca_loss, rec_loss, ccor, dist):
    with writer.as_default():
        tf.summary.scalar("Loss/Total", loss, step=epoch)
        tf.summary.scalar("Loss/CCA", cca_loss, step=epoch)
        tf.summary.scalar("Loss/reconstruction", rec_loss, step=epoch)
        tf.summary.scalar("Canonical correlation/0", ccor[0], step=epoch)
        tf.summary.scalar("Canonical correlation/1", ccor[1], step=epoch)
        tf.summary.scalar("Canonical correlation/2", ccor[2], step=epoch)
        tf.summary.scalar("Canonical correlation/3", ccor[3], step=epoch)
        tf.summary.scalar("Canonical correlation/4", ccor[4], step=epoch)
        tf.summary.scalar("Distance measure", dist, step=epoch)
        writer.flush()
