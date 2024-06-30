# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------

import tensorflow as tf

# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# https://github.com/sayakpaul/Adaptive-Gradient-Clipping
# https://arxiv.org/abs/2102.06171
# --------------------------------------------------------------------------------------------

def compute_norm(x, axis, keepdims):

    """
    Computes the norm of the input tensor along the specified axis.

    Args:
        x (tf.Tensor): Input tensor.
        axis (Union[int, List[int], Tuple[int]]): Axis or axes along which to compute the norm.
        keepdims (bool): Whether to keep the dimensions or not.

    Returns:
        tf.Tensor: Norm of the input tensor.
    """

    return tf.math.reduce_sum(x ** 2, axis=axis, keepdims=keepdims) ** 0.5

def unitwise_norm(x):

    """
    Computes the unit-wise norm of the input tensor.

    Args:
        x (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Unit-wise norm of the input tensor.
    """

    shape = len(x.get_shape())

    if shape <= 1:

        axis = None
        keepdims = False

    elif shape in [2, 3]:

        axis = 0
        keepdims = True

    elif shape == 4:

        axis = [0, 1, 2]
        keepdims = True

    else:

        raise ValueError(f"Got a parameter with shape not in [1, 2, 4]! {x}")

    return compute_norm(x, axis, keepdims)

def adaptive_clip_grad(parameters, gradients, clip_factor=0.01, eps=1e-3):

    """
    Applies adaptive gradient clipping to the gradients.

    Args:
        parameters (list of tf.Tensor): List of parameters.
        gradients (list of tf.Tensor): List of gradients.
        clip_factor (float): Clipping factor.
        eps (float): Small value to avoid division by zero.

    Returns:
        list of tf.Tensor: Clipped gradients.
    """

    new_grads = []

    for (params, grads) in zip(parameters, gradients):

        p_norm = unitwise_norm(params)
        max_norm = tf.math.maximum(p_norm, eps) * clip_factor
        grad_norm = unitwise_norm(grads)
        clipped_grad = grads * (max_norm / tf.math.maximum(grad_norm, 1e-6))
        new_grad = tf.where(grad_norm < max_norm, grads, clipped_grad)
        new_grads.append(new_grad)

    return new_grads