# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# https://github.com/yhenon/keras-spp
# https://arxiv.org/pdf/1406.4729.pdf
# --------------------------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package='SPP')
class SpatialPyramidPooling(layers.Layer):

    """
    Spatial Pyramid Pooling layer implementation based on the paper:
    "Spatial Pyramid Pooling in Deep Convolutional Networks for Image Classification" by K. He et al.
    
    Args:
        name (str): Name of the layer.
        pool_list (list): List of pooling sizes.
    """

    def __init__(self, name: str, pool_list: list, **kwargs):

        """
        Initializes the SpatialPyramidPooling layer.
        
        Args:
            name (str): Name of the layer.
            pool_list (list): List of pooling sizes.
        """

        super(SpatialPyramidPooling, self).__init__(**kwargs)

        self.name_layer = name
        self.pool_list = pool_list
        self.num_outputs_per_channel = sum([i * i for i in pool_list])
        self.norm = layers.LayerNormalization(name=self.name_layer + "_layernorm")
        self.activation = layers.Activation('gelu', name=self.name_layer + "_activation")

    def get_config(self) -> dict:

        """
        Returns the configuration of the layer.
        
        Returns:
            dict: Configuration of the layer.
        """

        return {'name': self.name_layer, 'pool_list': self.pool_list}

    def call(self, x: tf.Tensor) -> tf.Tensor:

        """
        Calls the SpatialPyramidPooling layer.
        
        Args:
            x (tf.Tensor): Input tensor to the layer.
        
        Returns:
            tf.Tensor: Output tensor of the layer.
        """

        input_shape = tf.shape(x)

        # Calculate the row and column lengths for each pooling size
        row_length = [tf.cast(input_shape[1], 'float32') / i for i in self.pool_list]
        col_length = [tf.cast(input_shape[2], 'float32') / i for i in self.pool_list]

        outputs = []

        # Iterate over each pooling size and calculate the pooling regions
        for pool_num, num_pool_regions in enumerate(self.pool_list):

            for jy in range(num_pool_regions):

                for ix in range(num_pool_regions):

                    x1 = ix * col_length[pool_num]
                    x2 = ix * col_length[pool_num] + col_length[pool_num]
                    y1 = jy * row_length[pool_num]
                    y2 = jy * row_length[pool_num] + row_length[pool_num]

                    # Convert the coordinates to integers
                    x1 = tf.cast(tf.round(x1), dtype=tf.int32)
                    x2 = tf.cast(tf.round(x2), dtype=tf.int32)
                    y1 = tf.cast(tf.round(y1), dtype=tf.int32)
                    y2 = tf.cast(tf.round(y2), dtype=tf.int32)

                    # Calculate the new shape for the pooling region
                    new_shape = [input_shape[0], y2 - y1, x2 - x1, input_shape[3]]

                    # Crop the input tensor to the pooling region
                    x_crop = x[:, y1:y2, x1:x2, :]
                    xm = tf.reshape(tensor=x_crop, shape=new_shape)

                    # Calculate the maximum value in the pooling region
                    pooled_val = tf.reduce_max(xm, axis=(1, 2))

                    # Append the pooled value to the outputs list
                    outputs.append(pooled_val)

        # Concatenate the outputs list along the channel axis
        outputs = tf.concat(outputs, axis=1)

        # Apply layer normalization and activation to the outputs
        return self.activation(self.norm(outputs))