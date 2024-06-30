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

@keras.saving.register_keras_serializable(package = 'BiFPN')
class BiFPN(layers.Layer):

    def __init__(self, name, W_bifpn = 64, EPSILON = 1e-8, kernel_size_convs = 3):
        
        super(BiFPN, self).__init__()
        
        # Parameters
        self.name_layer = name
        self.W_bifpn = W_bifpn
        self.EPSILON = EPSILON
        self.kernel_size_convs = kernel_size_convs

        # Activation function
        # In my case I have tried to use functions like GeLU or ReLU but I get numerical instability after a while. 
        # I have tried clipping the gradients and adding additional regularization techniques but they did not work.
        self.activation = tf.keras.activations.swish
        
        # LayerNormalization
        self.layernorms = [layers.LayerNormalization(name = self.name_layer + '_layernorm_' + str(i)) for i in range(4)]
        
        # Methods to apply up or down sampling
        self.upsamplings = [layers.Conv2DTranspose(filters = self.W_bifpn, kernel_size = 2, strides = 2,
                                                    padding = 'same', name = self.name_layer + '_upsampling_' + str(i)) for i in range(2)]
        self.downsamplings = [APSDownsampling(filtros = self.W_bifpn) for i in range(2)]

        # Adjust the number of channels
        self.conv1x1s = [layers.Conv2D(filters = self.W_bifpn, kernel_size = 1, use_bias = False, kernel_initializer = "he_normal",
                                       name = self.name_layer + '_conv1x1_' + str(i)) for i in range(3)]

        # Circular padding
        self.padding = CircularPad(padding = (1, 1, 1, 1))

        # Conv2D for BiFPN -> Feature extraction from the backbone
        self.conv2_td = layers.SeparableConv2D(filters = self.W_bifpn, kernel_size = kernel_size_convs, name = self.name_layer + "_conv2_td")
        self.conv1_out = layers.SeparableConv2D(filters = self.W_bifpn, kernel_size = kernel_size_convs, name = self.name_layer + "_conv1_out")
        self.conv2_out = layers.SeparableConv2D(filters = self.W_bifpn, kernel_size = kernel_size_convs, name = self.name_layer + "_conv2_out")
        self.conv3_out = layers.SeparableConv2D(filters = self.W_bifpn, kernel_size = kernel_size_convs, name = self.name_layer + "_conv3_out")

        # Weigths used for knowing the importance of each feature extracted from BiFPN
        self.w11 = self.add_weight(self.name_layer + "W11", shape = [1, 1, 1, 1], initializer = tf.initializers.he_normal(), trainable = True)
        self.w21 = self.add_weight(self.name_layer + "W21", shape = [1, 1, 1, 1], initializer = tf.initializers.he_normal(), trainable = True)
        self.w12 = self.add_weight(self.name_layer + "W12", shape = [1, 1, 1, 1], initializer = tf.initializers.he_normal(), trainable = True)
        self.w22 = self.add_weight(self.name_layer + "W22", shape = [1, 1, 1, 1], initializer = tf.initializers.he_normal(), trainable = True)
        self.w13 = self.add_weight(self.name_layer + "W13", shape = [1, 1, 1, 1], initializer = tf.initializers.he_normal(), trainable = True)
        self.w23 = self.add_weight(self.name_layer + "W23", shape = [1, 1, 1, 1], initializer = tf.initializers.he_normal(), trainable = True)
        self.w33 = self.add_weight(self.name_layer + "W33", shape = [1, 1, 1, 1], initializer = tf.initializers.he_normal(), trainable = True)
        self.w14 = self.add_weight(self.name_layer + "W14", shape = [1, 1, 1, 1], initializer = tf.initializers.he_normal(), trainable = True)
        self.w24 = self.add_weight(self.name_layer + "W24", shape = [1, 1, 1, 1], initializer = tf.initializers.he_normal(), trainable = True)

    def get_config(self):

        return {'name': self.name_layer, 'W_bifpn': self.W_bifpn, 'EPSILON': self.EPSILON, 'kernel_size_convs': self.kernel_size_convs}

    def call(self, p1, p2, p3):

        """
        In this example we have only used 3 levels of hierarchy in the BackBone used,
        this may vary depending on the depth of the model/requirement.
        """

        # Set all features to the same number of channels
        p1 = self.conv1x1s[0](p1)
        p2 = self.conv1x1s[1](p2)
        p3 = self.conv1x1s[2](p3)

        # P2 Intermediate
        temp = self.w11 * self.upsamplings[0](p3) + self.w21 * p2
        P_2_td = self.activation(self.layernorms[0](self.conv2_td(self.padding(temp/(self.w11 + self.w21 + self.EPSILON)))))
        
        # P1 Output
        temp = self.w12 * p1 + self.w22 * self.upsamplings[1](P_2_td)
        P_1_out = self.activation(self.layernorms[1](self.conv1_out(self.padding(temp / (self.w12 + self.w22 + self.EPSILON)))))
        
        # P2 Output
        temp = self.w13 * self.downsamplings[0](P_1_out) + self.w23 * P_2_td + self.w33 * p2
        P_2_out = self.activation(self.layernorms[2](self.conv2_out(self.padding(temp / (self.w13 + self.w23 + self.w33 + self.EPSILON)))))
        
        # P3 Output
        temp = self.w14 * p3 + self.w24 * self.downsamplings[1](P_2_out)
        P_3_out = self.activation(self.layernorms[3](self.conv3_out(self.padding(temp / (self.w14 + self.w24 + self.EPSILON)))))
        
        return P_1_out, P_2_out, P_3_out

@keras.saving.register_keras_serializable(package = 'WeightedSum')
class WeightedSum(layers.Layer):

    """
    Class that allows assigning extra weights to the output of each BiFPN level in order to
    obtain a weighted sum of the different distributions and model the importance of each level.
    """

    def __init__(self, num_outputs, **kwargs):

        super(WeightedSum, self).__init__(**kwargs)

        self.num_outputs = num_outputs

    def get_config(self):

        return {'num_outputs': self.num_outputs}

    def build(self, input_shape):

        # Initialize weights to 1/num_outputs so that initial sum is 1
        self.weights = self.add_weight(shape=(self.num_outputs,), initializer = 'ones', trainable = True)

    def call(self, inputs):

        if not isinstance(inputs, list):

            raise ValueError('A WeightedSum layer should be called on a list of inputs.')

        weighted_inputs = tf.stack(inputs) * tf.reshape(tf.nn.softmax(self.weights), [self.num_outputs, 1, 1])

        return tf.reduce_sum(weighted_inputs, axis = 0)