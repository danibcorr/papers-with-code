# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------

from tensorflow.keras import layers
from .APS import CircularPad, APSDownsampling
from .CBAM import CBAM
import tensorflow as tf
from tensorflow import keras

# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# https://arxiv.org/abs/1911.09070
# --------------------------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package='BiFPN')
class BiFPN(layers.Layer):

    """
    Bidirectional Feature Pyramid Network (BiFPN) Layer.
    This layer fuses multi-scale features using learned weights, upsampling, and downsampling.
    """

    def __init__(self, name: str, W_bifpn: int = 64, EPSILON: float = 1e-8, kernel_size_convs: int = 3):

        super(BiFPN, self).__init__()
        
        # Parameters
        self.name_layer = name
        self.W_bifpn = W_bifpn
        self.EPSILON = EPSILON
        self.kernel_size_convs = kernel_size_convs

        # Activation function (Swish is used to prevent numerical instability)
        self.activation = tf.keras.activations.swish
        
        # LayerNormalization instances
        self.layernorms = [layers.LayerNormalization(name=f'{self.name_layer}_layernorm_{i}') for i in range(4)]
        
        # Upsampling layers (using transposed convolutions)
        self.upsamplings = [
            layers.Conv2DTranspose(filters=self.W_bifpn, kernel_size=2, strides=2, padding='same', name=f'{self.name_layer}_upsampling_{i}')
            for i in range(2)
        ]

        # Downsampling layers (using Adaptive Polyphase Sampling)
        self.downsamplings = [APSDownsampling(name=f'{self.name_layer}_downsampling_{i}', filters=self.W_bifpn) for i in range(2)]

        # Adjust the number of channels with 1x1 convolutions
        self.conv1x1s = [
            layers.Conv2D(filters=self.W_bifpn, kernel_size=1, use_bias=False, kernel_initializer="he_normal", name=f'{self.name_layer}_conv1x1_{i}')
            for i in range(3)
        ]

        # Circular padding for spatial convolution
        self.padding = CircularPad(padding=(1, 1, 1, 1))

        # Separable convolutions for feature refinement
        self.conv2_td = layers.SeparableConv2D(filters=self.W_bifpn, kernel_size=kernel_size_convs, name=f'{self.name_layer}_conv2_td')
        self.conv1_out = layers.SeparableConv2D(filters=self.W_bifpn, kernel_size=kernel_size_convs, name=f'{self.name_layer}_conv1_out')
        self.conv2_out = layers.SeparableConv2D(filters=self.W_bifpn, kernel_size=kernel_size_convs, name=f'{self.name_layer}_conv2_out')
        self.conv3_out = layers.SeparableConv2D(filters=self.W_bifpn, kernel_size=kernel_size_convs, name=f'{self.name_layer}_conv3_out')

        # Weights for feature merging (learnable parameters)
        self.weights = {
            'w11': self.add_weight(f'{self.name_layer}_W11', shape=[1, 1, 1, 1], initializer=tf.initializers.he_normal(), trainable=True),
            'w21': self.add_weight(f'{self.name_layer}_W21', shape=[1, 1, 1, 1], initializer=tf.initializers.he_normal(), trainable=True),
            'w12': self.add_weight(f'{self.name_layer}_W12', shape=[1, 1, 1, 1], initializer=tf.initializers.he_normal(), trainable=True),
            'w22': self.add_weight(f'{self.name_layer}_W22', shape=[1, 1, 1, 1], initializer=tf.initializers.he_normal(), trainable=True),
            'w13': self.add_weight(f'{self.name_layer}_W13', shape=[1, 1, 1, 1], initializer=tf.initializers.he_normal(), trainable=True),
            'w23': self.add_weight(f'{self.name_layer}_W23', shape=[1, 1, 1, 1], initializer=tf.initializers.he_normal(), trainable=True),
            'w33': self.add_weight(f'{self.name_layer}_W33', shape=[1, 1, 1, 1], initializer=tf.initializers.he_normal(), trainable=True),
            'w14': self.add_weight(f'{self.name_layer}_W14', shape=[1, 1, 1, 1], initializer=tf.initializers.he_normal(), trainable=True),
            'w24': self.add_weight(f'{self.name_layer}_W24', shape=[1, 1, 1, 1], initializer=tf.initializers.he_normal(), trainable=True)
        }

    def get_config(self) -> dict:

        return {'name': self.name_layer, 'W_bifpn': self.W_bifpn, 'EPSILON': self.EPSILON, 'kernel_size_convs': self.kernel_size_convs}

    def call(self, p1: tf.Tensor, p2: tf.Tensor, p3: tf.Tensor) -> tuple:

        """
        Forward pass through the BiFPN layer.

        Args:
            p1, p2, p3: Feature maps from different levels of the backbone.

        Returns:
            P_1_out, P_2_out, P_3_out: Refined feature maps from BiFPN.
        """

        # Set all features to the same number of channels
        p1 = self.conv1x1s[0](p1)
        p2 = self.conv1x1s[1](p2)
        p3 = self.conv1x1s[2](p3)

        # P2 Intermediate (Top-Down Pathway)
        temp = self.weights['w11'] * self.upsamplings[0](p3) + self.weights['w21'] * p2
        P_2_td = self.activation(self.layernorms[0](self.conv2_td(self.padding(temp / (self.weights['w11'] + self.weights['w21'] + self.EPSILON)))))

        # P1 Output (Top-Down Pathway)
        temp = self.weights['w12'] * p1 + self.weights['w22'] * self.upsamplings[1](P_2_td)
        P_1_out = self.activation(self.layernorms[1](self.conv1_out(self.padding(temp / (self.weights['w12'] + self.weights['w22'] + self.EPSILON)))))

        # P2 Output (Bottom-Up Pathway)
        temp = self.weights['w13'] * self.downsamplings[0](P_1_out) + self.weights['w23'] * P_2_td + self.weights['w33'] * p2
        P_2_out = self.activation(self.layernorms[2](self.conv2_out(self.padding(temp / (self.weights['w13'] + self.weights['w23'] + self.weights['w33'] + self.EPSILON)))))

        # P3 Output (Bottom-Up Pathway)
        temp = self.weights['w14'] * p3 + self.weights['w24'] * self.downsamplings[1](P_2_out)
        P_3_out = self.activation(self.layernorms[3](self.conv3_out(self.padding(temp / (self.weights['w14'] + self.weights['w24'] + self.EPSILON)))))

        return P_1_out, P_2_out, P_3_out

@keras.saving.register_keras_serializable(package='WeightedSum')
class WeightedSum(layers.Layer):

    """
    A layer that computes a weighted sum of its inputs.
    The weights are trainable, allowing the model to learn the importance of each input.
    """

    def __init__(self, num_outputs: int, **kwargs):

        super(WeightedSum, self).__init__(**kwargs)

        self.num_outputs = num_outputs

    def get_config(self) -> dict:

        return {'num_outputs': self.num_outputs}

    def build(self, input_shape):

        # Initialize weights to 1/num_outputs so that initial sum is uniform
        self.weights = self.add_weight(shape=(self.num_outputs,), initializer='ones', trainable=True)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:

        if not isinstance(inputs, list):

            raise ValueError('A WeightedSum layer should be called on a list of inputs.')

        # Compute weighted sum of inputs
        weighted_inputs = tf.stack(inputs) * tf.reshape(tf.nn.softmax(self.weights), [self.num_outputs, 1, 1])

        return tf.reduce_sum(weighted_inputs, axis=0)