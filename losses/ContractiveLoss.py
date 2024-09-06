import tensorflow as tf

# --------------------------------------------------------------------------------------------
# https://arxiv.org/pdf/1412.5068v4.pdf
# --------------------------------------------------------------------------------------------


def contractive_loss(
    x: tf.Tensor, x_hat: tf.Tensor, encoder: tf.keras.Model, lam: float = 1e-4
) -> tf.Tensor:
    """
    This function computes the contractive loss for an autoencoder.

    Args:
        x (tf.Tensor): Original input tensor.
        x_hat (tf.Tensor): Reconstructed tensor.
        encoder (tf.keras.Model): Encoder part of the autoencoder.
        lam (float, optional): Regularization parameter. Defaults to 1e-4.

    Returns:
        Tensor: The contractive loss.
    """

    # Compute the reconstruction loss
    reconstruction_loss = tf.reduce_mean(tf.square(x - x_hat))

    # Compute the contractive loss
    contractive_loss = tf.reduce_sum(tf.square(tf.gradients(encoder(x), x)[0]))

    # Return the sum of the reconstruction loss and the contractive loss
    return reconstruction_loss + lam * contractive_loss
