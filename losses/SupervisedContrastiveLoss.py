# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------

import tensorflow as tf

# --------------------------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# https://arxiv.org/abs/2004.11362
# --------------------------------------------------------------------------------------------

@tf.function
def npairs_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:

    """
    This function computes the npairs loss between the true and predicted labels.

    Args:
        y_true (Tensor): True labels.
        y_pred (Tensor): Predicted labels.

    Returns:
        Tensor: The npairs loss.
    """

    # Convert the inputs to tensors
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Expand the dimensions of the true labels and compute a boolean mask of equal pairs
    y_true = tf.expand_dims(y_true, -1)
    y_true = tf.cast(tf.equal(y_true, tf.transpose(y_true)), y_pred.dtype)

    # Normalize the true labels
    y_true /= tf.math.reduce_sum(y_true, 1, keepdims = True)

    # Compute the softmax cross entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits = y_pred, labels = y_true)

    # Return the mean loss
    return tf.math.reduce_mean(loss)

class SupervisedContrastiveLoss(tf.keras.losses.Loss):

    """
    This class defines the Supervised Contrastive Learning loss.

    Attributes:
    - temperature (float): The temperature parameter for the contrastive loss. Defaults to 1.
    """

    def __init__(self, temperature: float = 1.0, name: str = None):

        super(SupervisedContrastiveLoss, self).__init__(name = name)

        self.temperature = temperature

    def call(self, labels: tf.Tensor, embeddings: tf.Tensor) -> tf.Tensor:

        """
        This method computes the SCL loss.

        Args:
            labels (Tensor): The true labels.
            embeddings (Tensor): The embeddings produced by the model.

        Returns:
            Tensor: The SCL loss.
        """

        # Normalize the embeddings
        embeddings = tf.math.l2_normalize(embeddings, axis = -1)

        # Compute the similarity matrix
        similarity_matrix = tf.matmul(embeddings, embeddings, transpose_b = True) / self.temperature

        # Compute and return the npairs loss
        return npairs_loss(tf.squeeze(labels), similarity_matrix)