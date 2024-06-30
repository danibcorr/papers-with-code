# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------

import tensorflow as tf

# --------------------------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# https://arxiv.org/pdf/1412.5068v4.pdf
# --------------------------------------------------------------------------------------------

def contractive_loss(x, x_hat, encoder, lam = 1e-4):

    """
    This function computes the contractive loss for an autoencoder.

    Parameters:
    - x (Tensor): Original input tensor.
    - x_hat (Tensor): Reconstructed tensor.
    - encoder (Model): Encoder part of the autoencoder.
    - lam (float, optional): Regularization parameter. Defaults to 1e-4.

    Returns:
    - Tensor: The contractive loss.
    """

    # Compute the reconstruction loss
    reconstruction_loss = tf.reduce_mean(tf.square(x - x_hat))

    # Compute the contractive loss
    contractive_loss = tf.reduce_sum(tf.square(tf.gradients(encoder(x), x)[0]))

    # Return the sum of the reconstruction loss and the contractive loss
    return reconstruction_loss + lam * contractive_loss
