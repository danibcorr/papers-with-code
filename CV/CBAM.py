# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# https://arxiv.org/pdf/1807.06521.pdf
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

@keras.saving.register_keras_serializable(package = 'MaxMinImportance')
class MaxMinImportance(layers.Layer):

    def __init__(self, name, **kwargs):

        super(MaxMinImportance, self).__init__(**kwargs)

        self.name_layer = name

        self.p0 = self.add_weight(self.name_layer + "p0", shape = (), initializer = 'ones', trainable = True, constraint = lambda x: tf.clip_by_value(x, 0, 1))
        self.p1 = self.add_weight(self.name_layer + "p1", shape = (), initializer = 'ones', trainable = True, constraint = lambda x: tf.clip_by_value(x, 0, 1))

    def get_config(self):

        return {'name': self.name_layer}

    def call(self, inputs):

        maxpool, minpool = inputs

        # Calcular la proporción cuadrada de self.p0 en relación con la suma de los cuadrados de self.p0 y self.p1.
        # Para calcular proporciones
        # Calcular lambda y 1 - lambda
        lambda_val = self.p0 ** 2 / (self.p0 ** 2 + self.p1 ** 2)
        one_minus_lambda = self.p1 ** 2 / (self.p0 ** 2 + self.p1 ** 2)

        return lambda_val * maxpool + one_minus_lambda * minpool

@keras.saving.register_keras_serializable(package = 'GlobalMinPooling2D')
class GlobalMinPooling2D(layers.Layer):

    def __init__(self, **kwargs):

        super(GlobalMinPooling2D, self).__init__(**kwargs)

    def call(self, inputs):

        return tf.reduce_min(inputs, axis = [1, 2])

@keras.saving.register_keras_serializable(package = 'ChannelAttentionModule')
class ChannelAttentionModule(layers.Layer):

    def __init__(self, name, use_min = False, ratio = 16):

        super(ChannelAttentionModule, self).__init__()

        self.name_layer = name
        self.use_min = use_min
        self.ratio = ratio

        self.l1 = None
        self.l2 = None

        self.variable = GlobalMinPooling2D(name = f"GMinP_CAM_{name}") if use_min else layers.GlobalAveragePooling2D(name = f"GAveP_CAM_{name}")
        self.gmp = layers.GlobalMaxPooling2D(name = f"GMaxP_CAM_{name}")
        self.mmi = MaxMinImportance(name = f"MMI_CAM_{name}")

        self.activation = layers.Activation('sigmoid', name = f"Activation_CAM_{name}")

    def get_config(self):

        return {'name': self.name_layer, 'use_min': self.use_min, 'ratio': self.ratio}

    def build(self, input_shape):

        channel = input_shape[-1]

        self.l1 = layers.Dense(channel // self.ratio, activation = 'relu', use_bias = False)
        self.l2 = layers.Dense(channel, use_bias = False)

    def call(self, inputs):

        variable_pool = self.l2(self.l1(self.variable(inputs)))

        maxpool = self.l2(self.l1(self.gmp(inputs)))

        concat = self.activation(self.mmi([maxpool, variable_pool]))

        return layers.Multiply()([inputs, concat])

@keras.saving.register_keras_serializable(package = 'SpatialAttentionModule')
class SpatialAttentionModule(layers.Layer):

    def __init__(self, name, use_min = False):

        super(SpatialAttentionModule, self).__init__()

        self.name_layer = name
        self.use_min = use_min

        self.padding = CircularPad((3, 3, 3, 3))
        self.conv = layers.Conv2D(1, kernel_size = 7, activation = 'sigmoid', name = f"Conv2D_SAM_{name}")

    def get_config(self):

        return {'name': self.name_layer, 'use_min': use_min}

    def call(self, inputs):

        variable_pool = tf.reduce_min(inputs, axis = -1) if use_min else tf.reduce_mean(inputs, axis = -1)
        variable_pool = tf.expand_dims(variable_pool, axis = -1)

        maxpool = tf.reduce_max(inputs, axis = -1)
        maxpool = tf.expand_dims(maxpool, axis = -1)

        concat = layers.Concatenate()([variable_pool, maxpool])

        conv = self.conv(self.padding(concat))

        return layers.Multiply()([inputs, conv])

@keras.saving.register_keras_serializable(package = 'CBAM')
class CBAM(layers.Layer):

    def __init__(self, name, use_min):

        super(CBAM, self).__init__()
        
        self.name_layer = name
        self.use_min = use_min

        self.channel_attention = ChannelAttentionModule(name, use_min)
        self.spatial_attention = SpatialAttentionModule(name, use_min)

    def get_config(self):

        return {'name': self.name_layer, 'use_min': self.use_min}

    def call(self, inputs):

        local_channel_att = self.channel_attention(inputs)
        local_spatial_att = self.spatial_attention(inputs)
        local_att = (local_channel_att * local_spatial_att) + local_channel_att  

        local_att = tf.expand_dims(local_att, axis = -1) 
        x = tf.expand_dims(inputs, axis = -1) 

        all_feature_maps = tf.concat([local_att, x], axis = -1)

        weights = tf.reshape(tf.nn.softmax(self.fusion_weights, axis = -1), (1, 1, 1, 1, 2))
        fused_feature_maps = tf.reduce_sum(all_feature_maps * weights, axis = -1)

        return fused_feature_maps