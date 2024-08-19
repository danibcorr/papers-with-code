# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# https://arxiv.org/pdf/1709.01507.pdf
# --------------------------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package='SE')
class SqueezeAndExcitation(layers.Layer):

    """
    Squeeze and Excitation layer implementation based on the paper:
    "Squeeze-and-Excitation Networks" by Jie Hu et al.
    
    Args:
        name (str): Name of the layer.
        num_filters (int): Number of filters in the layer.
        expansion (float, optional): Expansion factor for the dense layers. Defaults to 0.25.
    """

    def __init__(self, name: str, num_filters: int, expansion: float = 0.25):

        """
        Initializes the SqueezeAndExcitation layer.
        
        Args:
            name (str): Name of the layer.
            num_filters (int): Number of filters in the layer.
            expansion (float, optional): Expansion factor for the dense layers. Defaults to 0.25.
        """
        
        super(SqueezeAndExcitation, self).__init__()
        
        self.name_layer = name
        self.num_filters = num_filters
        self.expansion = expansion
        
        # Define the sequential layers for the Squeeze and Excitation mechanism
        self.layers = tf.keras.Sequential([
            # Global Average Pooling 2D layer to reduce spatial dimensions
            layers.GlobalAvgPool2D(keepdims=True, name=self.name_layer + "_se_gap_2d"),
            # Dense layer with GELU activation to learn the channel-wise attention
            layers.Dense(int(num_filters * expansion), use_bias=False, activation='gelu', name=self.name_layer + "_se_dense_gelu"),
            # Dense layer with sigmoid activation to produce the final attention weights
            layers.Dense(num_filters, use_bias=False, activation='sigmoid', name=self.name_layer + "_se_dense_sigmoid")
        ])

    def get_config(self) -> dict:
        
        """
        Returns the configuration of the layer.
        
        Returns:
            dict: Configuration of the layer.
        """

        return {'name': self.name_layer, 'num_filters': self.num_filters, 'expansion': self.expansion}

    def call(self, inputs: tf.Tensor) -> tf.Tensor:

        """
        Calls the Squeeze and Excitation layer.
        
        Args:
            inputs (tf.Tensor): Input tensor to the layer.
        
        Returns:
            tf.Tensor: Output tensor of the layer.
        """

        # Apply the Squeeze and Excitation mechanism to the input tensor
        return self.layers(inputs) * inputs