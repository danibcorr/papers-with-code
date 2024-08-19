# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------

import tensorflow as tf

# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# https://keras.io/examples/vision/gradient_centralization/
# https://arxiv.org/abs/2004.01461
# --------------------------------------------------------------------------------------------

class GCAdamW(tf.keras.optimizers.AdamW):

    """
    Gradient Centralization Optimizer for AdamW.
    """

    def __init__(self, name: str = "GCAdamW", **kwargs):

        """
        Initializes the Gradient Centralization Optimizer.

        Args:
            name (str): Name of the optimizer.
            **kwargs: Additional arguments.
        """

        super(GCAdamW, self).__init__(name, **kwargs)

    def get_gradients(self, loss: tf.Tensor, params: list or tf.Tensor) -> list or tf.Tensor:

        """
        Computes the gradients with gradient centralization.

        Args:
            loss (tf.Tensor): Loss tensor.
            params (list of tf.Tensor): List of parameters.

        Returns:
            list of tf.Tensor: List of gradients.
        """

        grads = []
        gradients = super().get_gradients(loss, params)

        for grad in gradients:

            grad_len = len(grad.shape)

            if grad_len > 1:

                axis = list(range(grad_len - 1))
                grad -= tf.math.reduce_mean(grad, axis=axis, keepdims=True)

            grads.append(grad)

        return grads