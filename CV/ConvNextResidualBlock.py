# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------

from tensorflow.keras import layers
from tensorflow import keras
from .SE import SqueezeAndExcitation
from .CBAM import CBAM
import tensorflow as tf
import tensorflow_probability as tfp

# --------------------------------------------------------------------------------------------
# KERNEL AND BIAS INITIALIZATION
# --------------------------------------------------------------------------------------------

kernel_initial = tf.keras.initializers.TruncatedNormal(stddev=0.2, seed = 42)
bias_initial = tf.keras.initializers.Constant(value=0)

# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# https://arxiv.org/pdf/2201.03545.pdf
# https://keras.io/api/keras_cv/layers/regularization/stochastic_depth/
# -------------------------------------------------------------------------------------------- 

class CircularPad(layers.Layer):

    """
    A custom layer to apply circular padding to the input tensor.
    """

    def __init__(self, padding = (1, 1, 1, 1)):

        """
        Initializes the CircularPad layer.

        Args:
            padding (tuple): Tuple specifying the padding sizes in the order (top, bottom, left, right).
        """

        super(CircularPad, self).__init__()

        self.pad_sizes = padding

    def call(self, x):

        """
        Applies circular padding to the input tensor.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Padded tensor.
        """

        top_pad, bottom_pad, left_pad, right_pad = self.pad_sizes

        # Circular padding for height dimension
        height_pad = tf.concat([x[:, -bottom_pad:], x, x[:, :top_pad]], axis=1)

        # Circular padding for width dimension
        return tf.concat([height_pad[:, :, -right_pad:], height_pad, height_pad[:, :, :left_pad]], axis=2)

class StochasticDepthResidual(layers.Layer):

    def __init__(self, rate = 0.25, **kwargs):

        super().__init__(**kwargs)
        self.rate = rate
        self.survival_probability = 1.0 - self.rate

    def call(self, x, training=None):

        if len(x) != 2:

            raise ValueError(f"""Input must be a list of length 2, got input with length={len(x)}.""")

        shortcut, residual = x

        b_l = keras.backend.random_bernoulli([], p=self.survival_probability)

        return shortcut + b_l * residual if training else shortcut + self.survival_probability * residual

    def get_config(self):

        config = {"rate": self.rate}
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))

@keras.saving.register_keras_serializable(package = 'ResidualBlock')
class ResidualBlock(layers.Layer):
    
    def __init__(self, name, num_filters, drop_prob = 0.25, layer_scale_init_value = 1e-6, use_cbam = False):
        
        super(ResidualBlock, self).__init__()
        
        # Parameters
        self.name_layer = name
        self.num_filters = num_filters
        self.drop_prob = drop_prob
        self.layer_scale_init_value = layer_scale_init_value
        self.use_cbam = use_cbam
        
        # We can use SE o CBAM, CBAM can be seen as an enhanced version of SE that tends to be more GPU friendly
        # and that not only models attention at the channel level but also at the spatial level.
        self.attention = SqueezeAndExcitation(name = self.name_layer + f"_se_input", num_filters = self.num_filters) if not use_cbam else CBAM(name = self.name_layer + "_CBAM")

        # Feature extraction
        self.layers = tf.keras.Sequential([
            CircularPad(padding = (1, 1, 1, 1)),
            layers.Conv2D(self.num_filters, kernel_size = 7, groups = num_filters, kernel_initializer = kernel_initial,
                          bias_initializer = bias_initial, name = self.name_layer + "_conv2d_7"),
            layers.LayerNormalization(name = self.name_layer + "_layernorm"),

            CircularPad(padding = (1, 1, 1, 1)),
            layers.Conv2D(self.num_filters * 4, kernel_size = 1, kernel_initializer = kernel_initial, 
                          bias_initializer = bias_initial, name = self.name_layer + "_conv2d_4"),
            layers.Activation('gelu', name = self.name_layer + "_activation"),

            CircularPad(padding = (1, 1, 1, 1)),
            layers.Conv2D(self.num_filters, kernel_size = 1, kernel_initializer = kernel_initial, 
                          bias_initializer = bias_initial, name = self.name_layer + "_conv2d_output")
        ], name = f"Sequential_Residual_{self.name_layer}")
        
        self.layer_scale_gamma = None
        
        if self.layer_scale_init_value > 0:
        
            with tf.init_scope():
                
                self.layer_scale_gamma = tf.Variable(name = self.name_layer + "_gamma", initial_value=self.layer_scale_init_value * tf.ones((self.num_filters)))

        self.stochastic_depth = StochasticDepthResidual(self.drop_prob)

    def get_config(self):

        return {'name': self.name_layer, 'num_filters': self.num_filters, 'drop_prob': self.drop_prob, 'layer_scale_init_value': self.layer_scale_init_value,
                'use_cbam': self.use_cbam}

    def call(self, inputs):
        
        # Feature extraction inputs
        x = self.layers(inputs)

        if self.layer_scale_gamma is not None:
            
            x = x * self.layer_scale_gamma
        
        # SE or CBAM block
        x = self.attention(x)

        # Residual + Regularization 
        x = self.stochastic_depth([inputs, x])
        
        return x