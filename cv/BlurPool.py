import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --------------------------------------------------------------------------------------------
# Reference: https://github.com/csvance/blur-pool-keras
# Paper: https://arxiv.org/abs/1904.11486
# --------------------------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="MaxBlurPooling2D")
class MaxBlurPooling2D(layers.Layer):
    """
    MaxBlurPooling2D Layer: Combines max pooling with a Gaussian blur to reduce aliasing.
    """

    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        """
        Initialize the MaxBlurPooling2D layer.

        Args:
            pool_size (int): Size of the pooling window.
            kernel_size (int): Size of the blur kernel (3 or 5).
            **kwargs: Additional keyword arguments for the parent class.
        """

        super(MaxBlurPooling2D, self).__init__(**kwargs)

        self.pool_size = pool_size
        self.kernel_size = kernel_size

    def get_config(self) -> dict:
        """
        Get the configuration of the layer.

        Returns:
            dict: Configuration dictionary.
        """

        return {"pool_size": self.pool_size, "kernel_size": self.kernel_size}

    def build(self, input_shape):
        """
        Build the layer.

        Args:
            input_shape (tuple): Shape of the input tensor.

        Raises:
            ValueError: If kernel_size is not 3 or 5.
        """

        if self.kernel_size == 3:

            bk = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0

        elif self.kernel_size == 5:

            bk = (
                np.array(
                    [
                        [1, 4, 6, 4, 1],
                        [4, 16, 24, 16, 4],
                        [6, 24, 36, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1, 4, 6, 4, 1],
                    ]
                )
                / 256.0
            )
        else:

            raise ValueError(
                f"Unsupported kernel_size {self.kernel_size}. Supported sizes: 3, 5."
            )

        bk = np.repeat(bk, input_shape[-1])
        bk = np.reshape(bk, (self.kernel_size, self.kernel_size, input_shape[-1], 1))
        blur_init = tf.keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(
            name="blur_kernel",
            shape=(self.kernel_size, self.kernel_size, input_shape[-1], 1),
            initializer=blur_init,
            trainable=False,
        )

        super(MaxBlurPooling2D, self).build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass of the layer.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor after max pooling and blurring.
        """

        x = tf.nn.pool(
            x,
            window_shape=(self.pool_size, self.pool_size),
            strides=(1, 1),
            padding="SAME",
            pooling_type="MAX",
            data_format="NHWC",
        )
        x = tf.nn.depthwise_conv2d(
            x,
            self.blur_kernel,
            strides=(self.pool_size, self.pool_size),
            padding="SAME",
        )

        return x

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """
        Compute the output shape of the layer.

        Args:
            input_shape (tuple): Shape of the input tensor.

        Returns:
            tuple: Shape of the output tensor.
        """

        return (
            input_shape[0],
            int(np.ceil(input_shape[1] / self.pool_size)),
            int(np.ceil(input_shape[2] / self.pool_size)),
            input_shape[3],
        )


@keras.saving.register_keras_serializable(package="AverageBlurPooling2D")
class AverageBlurPooling2D(layers.Layer):
    """
    AverageBlurPooling2D Layer: Combines average pooling with a Gaussian blur to reduce aliasing.
    """

    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        """
        Initialize the AverageBlurPooling2D layer.

        Args:
            pool_size (int): Size of the pooling window.
            kernel_size (int): Size of the blur kernel (3 or 5).
            **kwargs: Additional keyword arguments for the parent class.
        """

        super(AverageBlurPooling2D, self).__init__(**kwargs)

        self.pool_size = pool_size
        self.kernel_size = kernel_size

    def get_config(self) -> dict:
        """
        Get the configuration of the layer.

        Returns:
            dict: Configuration dictionary.
        """

        return {"pool_size": self.pool_size, "kernel_size": self.kernel_size}

    def build(self, input_shape):
        """
        Build the layer.

        Args:
            input_shape (tuple): Shape of the input tensor.

        Raises:
            ValueError: If kernel_size is not 3 or 5.
        """

        if self.kernel_size == 3:

            bk = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0

        elif self.kernel_size == 5:

            bk = (
                np.array(
                    [
                        [1, 4, 6, 4, 1],
                        [4, 16, 24, 16, 4],
                        [6, 24, 36, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1, 4, 6, 4, 1],
                    ]
                )
                / 256.0
            )

        else:

            raise ValueError(
                f"Unsupported kernel_size {self.kernel_size}. Supported sizes: 3, 5."
            )

        bk = np.repeat(bk, input_shape[-1])
        bk = np.reshape(bk, (self.kernel_size, self.kernel_size, input_shape[-1], 1))
        blur_init = tf.keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(
            name="blur_kernel",
            shape=(self.kernel_size, self.kernel_size, input_shape[-1], 1),
            initializer=blur_init,
            trainable=False,
        )

        super(AverageBlurPooling2D, self).build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass of the layer.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor after average pooling and blurring.
        """

        x = tf.nn.pool(
            x,
            window_shape=(self.pool_size, self.pool_size),
            strides=(1, 1),
            padding="SAME",
            pooling_type="AVG",
            data_format="NHWC",
        )
        x = tf.nn.depthwise_conv2d(
            x,
            self.blur_kernel,
            strides=(self.pool_size, self.pool_size),
            padding="SAME",
        )

        return x

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """
        Compute the output shape of the layer.

        Args:
            input_shape (tuple): Shape of the input tensor.

        Returns:
            tuple: Shape of the output tensor.
        """

        return (
            input_shape[0],
            int(np.ceil(input_shape[1] / self.pool_size)),
            int(np.ceil(input_shape[2] / self.pool_size)),
            input_shape[3],
        )


@keras.saving.register_keras_serializable(package="BlurPool2D")
class BlurPool2D(layers.Layer):
    """
    BlurPool2D Layer: Applies a Gaussian blur followed by downsampling to reduce aliasing.
    """

    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        """
        Initialize the BlurPool2D layer.

        Args:
            pool_size (int): Size of the pooling window.
            kernel_size (int): Size of the blur kernel (3 or 5).
            **kwargs: Additional keyword arguments for the parent class.
        """

        super(BlurPool2D, self).__init__(**kwargs)

        self.pool_size = pool_size
        self.kernel_size = kernel_size

    def get_config(self) -> dict:
        """
        Get the configuration of the layer.

        Returns:
            dict: Configuration dictionary.
        """

        return {"pool_size": self.pool_size, "kernel_size": self.kernel_size}

    def build(self, input_shape):
        """
        Build the layer.

        Args:
            input_shape (tuple): Shape of the input tensor.

        Raises:
            ValueError: If kernel_size is not 3 or 5.
        """

        if self.kernel_size == 3:

            bk = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0

        elif self.kernel_size == 5:

            bk = (
                np.array(
                    [
                        [1, 4, 6, 4, 1],
                        [4, 16, 24, 16, 4],
                        [6, 24, 36, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1, 4, 6, 4, 1],
                    ]
                )
                / 256.0
            )

        else:

            raise ValueError(
                f"Unsupported kernel_size {self.kernel_size}. Supported sizes: 3, 5."
            )

        bk = np.repeat(bk, input_shape[-1])
        bk = np.reshape(bk, (self.kernel_size, self.kernel_size, input_shape[-1], 1))
        blur_init = tf.keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(
            name="blur_kernel",
            shape=(self.kernel_size, self.kernel_size, input_shape[-1], 1),
            initializer=blur_init,
            trainable=False,
        )

        super(BlurPool2D, self).build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass of the layer.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor after blurring and downsampling.
        """

        return tf.nn.depthwise_conv2d(
            x,
            self.blur_kernel,
            strides=(self.pool_size, self.pool_size),
            padding="SAME",
        )

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """
        Compute the output shape of the layer.

        Args:
            input_shape (tuple): Shape of the input tensor.

        Returns:
            tuple: Shape of the output tensor.
        """

        return (
            input_shape[0],
            int(np.ceil(input_shape[1] / self.pool_size)),
            int(np.ceil(input_shape[2] / self.pool_size)),
            input_shape[3],
        )
