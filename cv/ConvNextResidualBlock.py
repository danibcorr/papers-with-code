# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------

from tensorflow.keras import layers
from tensorflow import keras
from .APS import CircularPad
from .SE import SqueezeAndExcitation
from .CBAM import CBAM
import tensorflow as tf
import tensorflow_probability as tfp

# --------------------------------------------------------------------------------------------
# KERNEL AND BIAS INITIALIZATION
# --------------------------------------------------------------------------------------------

kernel_initial = tf.keras.initializers.TruncatedNormal(stddev=0.2, seed=42)
bias_initial = tf.keras.initializers.Constant(value=0)

# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# References:
# https://arxiv.org/pdf/2201.03545.pdf
# https://keras.io/api/keras_cv/layers/regularization/stochastic_depth/
# -------------------------------------------------------------------------------------------- 

@keras.saving.register_keras_serializable(package='StochasticDepthResidual')
class StochasticDepthResidual(layers.Layer):

   """
   Implements Stochastic Depth for residual connections.
   """

   def __init__(self, rate: float = 0.25, **kwargs):

       """
       Initializes the StochasticDepthResidual layer.

       Args:
           rate (float): Drop rate for stochastic depth.
           **kwargs: Additional keyword arguments for the parent class.
       """
       super(StochasticDepthResidual, self).__init__(**kwargs)

       self.rate = rate
       self.survival_probability = 1.0 - self.rate

   def call(self, x: list, training: bool = None) -> tf.Tensor:

       """
       Applies stochastic depth to the residual connection.

       Args:
           x (list): List containing [shortcut, residual].
           training (bool): Whether in training mode or not.

       Returns:
           tf.Tensor: Output after applying stochastic depth.

       Raises:
           ValueError: If input is not a list of length 2.
       """

       if len(x) != 2:

           raise ValueError(f"Input must be a list of length 2, got input with length={len(x)}.")

       shortcut, residual = x

       b_l = keras.backend.random_bernoulli([], p=self.survival_probability)

       return shortcut + b_l * residual if training else shortcut + self.survival_probability * residual

   def get_config(self) -> dict:

       """
       Returns the configuration of the layer.

       Returns:
           dict: Configuration dictionary.
       """

       config = {"rate": self.rate}
       base_config = super().get_config()

       return dict(list(base_config.items()) + list(config.items()))

@keras.saving.register_keras_serializable(package='ResidualBlock')
class ResidualBlock(layers.Layer):

   """
   Implements a Residual Block with optional attention mechanisms.
   """
   
   def __init__(self, name: str, num_filters: int, drop_prob: float = 0.25, 
                layer_scale_init_value: float = 1e-6, use_cbam: bool = False):

       """
       Initializes the ResidualBlock.

       Args:
           name (str): Name of the layer.
           num_filters (int): Number of filters in the convolutional layers.
           drop_prob (float): Drop probability for stochastic depth.
           layer_scale_init_value (float): Initial value for layer scaling.
           use_cbam (bool): Whether to use CBAM attention or SE attention.
       """

       super(ResidualBlock, self).__init__()
       
       self.name_layer = name
       self.num_filters = num_filters
       self.drop_prob = drop_prob
       self.layer_scale_init_value = layer_scale_init_value
       self.use_cbam = use_cbam
       
       self.attention = (CBAM(name=f"{self.name_layer}_CBAM") if use_cbam 
                         else SqueezeAndExcitation(name=f"{self.name_layer}_se_input", num_filters=self.num_filters))

       self.layers = tf.keras.Sequential([
           CircularPad(padding=(1, 1, 1, 1)),
           layers.Conv2D(self.num_filters, kernel_size=7, groups=num_filters, 
                         kernel_initializer=kernel_initial, bias_initializer=bias_initial, 
                         name=f"{self.name_layer}_conv2d_7"),
           layers.LayerNormalization(name=f"{self.name_layer}_layernorm"),
           CircularPad(padding=(1, 1, 1, 1)),
           layers.Conv2D(self.num_filters * 4, kernel_size=1, kernel_initializer=kernel_initial, 
                         bias_initializer=bias_initial, name=f"{self.name_layer}_conv2d_4"),
           layers.Activation('gelu', name=f"{self.name_layer}_activation"),
           CircularPad(padding=(1, 1, 1, 1)),
           layers.Conv2D(self.num_filters, kernel_size=1, kernel_initializer=kernel_initial, 
                         bias_initializer=bias_initial, name=f"{self.name_layer}_conv2d_output")
       ], name=f"Sequential_Residual_{self.name_layer}")
       
       self.layer_scale_gamma = None
       
       if self.layer_scale_init_value > 0:

           with tf.init_scope():

               self.layer_scale_gamma = tf.Variable(
                   name=f"{self.name_layer}_gamma", 
                   initial_value=self.layer_scale_init_value * tf.ones((self.num_filters))
               )

       self.stochastic_depth = StochasticDepthResidual(self.drop_prob)

   def get_config(self) -> dict:

       """
       Returns the configuration of the layer.

       Returns:
           dict: Configuration dictionary.
       """

       return {
           'name': self.name_layer, 
           'num_filters': self.num_filters, 
           'drop_prob': self.drop_prob, 
           'layer_scale_init_value': self.layer_scale_init_value,
           'use_cbam': self.use_cbam
       }

   def call(self, inputs: tf.Tensor) -> tf.Tensor:

       """
       Forward pass of the ResidualBlock.

       Args:
           inputs (tf.Tensor): Input tensor.

       Returns:
           tf.Tensor: Output tensor after applying the residual block.
       """

       x = self.layers(inputs)

       if self.layer_scale_gamma is not None:

           x = x * self.layer_scale_gamma
       
       x = self.attention(x)
       x = self.stochastic_depth([inputs, x])
       
       return x