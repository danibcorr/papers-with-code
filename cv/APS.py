# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# https://github.com/nicholausdy/Adaptive-Polyphase-Sampling-Keras
# https://arxiv.org/pdf/2011.14214.pdf
# --------------------------------------------------------------------------------------------

class CircularPad(layers.Layer):

    """
    A custom Keras layer to apply circular padding to the input tensor.
    """

    def __init__(self, padding: tuple = (1, 1, 1, 1)):

        """
        Initializes the CircularPad layer.

        Args:
            padding (tuple): Tuple specifying the padding sizes in the order (top, bottom, left, right).
        """

        super(CircularPad, self).__init__()

        self.pad_sizes = padding

    def call(self, x: tf.Tensor) -> tf.Tensor:

        """
        Applies circular padding to the input tensor.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Padded tensor.
        """

        top_pad, bottom_pad, left_pad, right_pad = self.pad_sizes

        # Circular padding for the height dimension
        height_pad = tf.concat([x[:, -bottom_pad:], x, x[:, :top_pad]], axis=1)

        # Circular padding for the width dimension
        return tf.concat([height_pad[:, :, -right_pad:], height_pad, height_pad[:, :, :left_pad]], axis=2)


@keras.saving.register_keras_serializable(package='APSLayer')
class APSLayer(layers.Layer):

    """
    Adaptive Polyphase Sampling (APS) Layer.
    This layer extracts polyphase components from the input and returns the component
    with the maximum norm and its corresponding index.
    """

    def __init__(self, stride: int = 2, order: int = 2, name: str = None, **kwargs):

        """
        Initializes the APSLayer.

        Args:
            stride (int): The stride used for downsampling. Default is 2.
            order (int): The order of the norm to use when comparing polyphase components. Default is 2.
            name (str, optional): The name of the layer.
        """

        super(APSLayer, self).__init__(**kwargs)

        self.stride = stride
        self.order = order
        self.name_layer = name

    def call(self, inputs: tf.Tensor) -> tuple:

        """
        Applies the APS algorithm to downsample the input.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: The downsampled tensor and the index of the max-norm component.
        """

        downsampled, max_norm_index = self.downsample(inputs)

        return downsampled, max_norm_index

    def get_config(self) -> dict:

        """
        Returns the configuration of the layer for serialization.

        Returns:
            dict: Configuration dictionary.
        """

        return {'name': self.name_layer, "stride": self.stride, "order": self.order}

    @tf.function
    def downsample(self, inputs: tf.Tensor) -> tuple:

        """
        Performs the downsampling by selecting the polyphase component with the maximum norm.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: The selected polyphase component and its index.
        """

        polyphase_components = tf.TensorArray(tf.float32, size=self.stride ** 2, dynamic_size=False, clear_after_read=False)
        input_shape = tf.shape(inputs)

        # Gather polyphase components
        for arr_index, (i, j) in enumerate([(i, j) for i in range(self.stride) for j in range(self.stride)]):

            strided_matrix = tf.strided_slice(
                inputs,
                begin=[0, i, j, 0],
                end=[input_shape[0], input_shape[1], input_shape[2], input_shape[3]],
                strides=[1, self.stride, self.stride, 1],
                begin_mask=9,
                end_mask=9
            )

            strided_matrix = tf.cast(strided_matrix, dtype=tf.float32)
            polyphase_components = polyphase_components.write(arr_index, strided_matrix)

        # Compute norms and find the index of the component with the maximum norm
        norms = tf.map_fn(
            lambda idx: tf.norm(tensor=polyphase_components.read(idx), ord=self.order),
            tf.range(self.stride ** 2),
            fn_output_signature=tf.float32
        )

        max_norm_index = tf.math.argmax(norms)
        max_norm_index = tf.cast(max_norm_index, dtype=tf.int32)

        # Return the component with the maximum norm and its index
        return polyphase_components.read(max_norm_index), max_norm_index


@keras.saving.register_keras_serializable(package='APSDownsampleGivenPolyIndices')
class APSDownsampleGivenPolyIndices(layers.Layer):
    
    """
    A layer that downsamples an input tensor given the indices of the polyphase component.
    """

    def __init__(self, stride: int = 2, name: str = None, **kwargs):

        """
        Initializes the APSDownsampleGivenPolyIndices layer.

        Args:
            stride (int): The stride used for downsampling. Default is 2.
            name (str, optional): The name of the layer.
        """

        super(APSDownsampleGivenPolyIndices, self).__init__(**kwargs)

        self.stride = stride
        self.name_layer = name

    def call(self, inputs: tf.Tensor, max_poly_indices: tf.Tensor) -> tf.Tensor:

        """
        Downsamples the input tensor based on the given polyphase indices.

        Args:
            inputs (tf.Tensor): Input tensor.
            max_poly_indices (tf.Tensor): Indices of the maximum polyphase components.

        Returns:
            tf.Tensor: The downsampled tensor.
        """

        return self.downsample(inputs, max_poly_indices)

    @tf.function(jit_compile=True)
    def downsample(self, inputs: tf.Tensor, max_poly_indices: tf.Tensor) -> tf.Tensor:

        """
        Performs the actual downsampling based on the maximum polyphase indices.

        Args:
            inputs (tf.Tensor): Input tensor.
            max_poly_indices (tf.Tensor): Indices of the maximum polyphase components.

        Returns:
            tf.Tensor: The downsampled tensor.
        """

        # Create index matrices for 'i' and 'j' positions
        i, j = tf.meshgrid(tf.range(self.stride), tf.range(self.stride), indexing='ij')

        # Flatten and stack the indices into a 4D tensor
        elem = tf.stack([tf.zeros_like(i), i, j, tf.zeros_like(i)], axis=-1)
        elem = tf.reshape(elem, [-1, 4])

        # Create a TensorArray for lookups and retrieve the appropriate indices
        lookup = tf.TensorArray(tf.int32, size=self.stride ** 2, clear_after_read=True)
        lookup = lookup.unstack(elem)
        max_poly_indices = lookup.read(max_poly_indices)

        # Obtain the shape of the input tensor
        input_shape = tf.shape(inputs)

        # Create the downsampled tensor using strided slicing
        strided_matrix = tf.strided_slice(
            inputs,
            begin=max_poly_indices,
            end=[input_shape[0], input_shape[1], input_shape[2], input_shape[3]],
            strides=[1, self.stride, self.stride, 1],
            begin_mask=9,
            end_mask=9
        )

        # Return the downsampled tensor
        return tf.cast(strided_matrix, dtype=tf.float32)

    def get_config(self) -> dict:

        """
        Returns the configuration of the layer for serialization.

        Returns:
            dict: Configuration dictionary.
        """

        return {'name': self.name_layer, "stride": self.stride}


@keras.saving.register_keras_serializable(package='APSDownsampling')
class APSDownsampling(layers.Layer):

    """
    A composite layer for adaptive polyphase downsampling.
    This layer combines circular padding, APSLayer, APSDownsampleGivenPolyIndices,
    and convolution layers to perform adaptive downsampling.
    """

    def __init__(self, name: str, filters: int):

        """
        Initializes the APSDownsampling layer.

        Args:
            name (str): The name of the layer.
            filters (int): The number of filters for the convolution layers.
        """

        super(APSDownsampling, self).__init__()

        self.name_layer = name
        self.filters = filters

        # Define the internal layers
        self.padding = CircularPad((1, 1, 1, 1))
        self.aps_layer = APSLayer(name=name + "APS_Layer")
        self.downsampling = APSDownsampleGivenPolyIndices(name=name + "APSDownsampleGivenPolyIndices_Layer")

        self.conv_y = layers.Conv2D(kernel_size=3, strides=1, filters=filters, name=name + "APS_conv_y")
        self.norm_y = layers.LayerNormalization(name=name + "APS_norm_y")
        self.activation_y = layers.Activation('gelu', name=name + "APS_act_y")

        self.conv_x = layers.Conv2D(kernel_size=3, strides=1, filters=filters, name=name + "APS_conv_x")
        self.norm_x = layers.LayerNormalization(name=name + "APS_norm_x")
        self.activation_x = layers.Activation('gelu', name=name + "APS_act_x")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:

        """
        Applies adaptive polyphase downsampling to the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor after downsampling and processing.
        """

        # Apply APSLayer to get downsampled tensor 'y' and the max-norm index
        y, max_norm_index = self.aps_layer(inputs)
        y = self.norm_y(y)
        y = self.activation_y(y)
        y = self.conv_y(self.padding(y))

        # Apply downsampling using the max-norm index to get tensor 'x'
        x = self.downsampling(inputs, max_norm_index)
        x = self.conv_x(self.padding(x))

        # Combine the results and apply normalization and activation
        x = layers.Add()([x, y])
        x = self.norm_x(x)
        x = self.activation_x(x)

        return x

    def get_config(self) -> dict:

        """
        Returns the configuration of the layer for serialization.

        Returns:
            dict: Configuration dictionary.
        """

        return {'name': self.name_layer, "filters": self.filters}