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

@keras.saving.register_keras_serializable(package = 'SPP')
class SpatialPyramidPooling(layers.Layer):

    def __init__(self, name, pool_list, **kwargs):

        super(SpatialPyramidPooling, self).__init__(**kwargs)

        self.name_layer = name
        self.pool_list = pool_list
        self.num_outputs_per_channel = sum([i * i for i in pool_list])
        self.norm = layers.LayerNormalization(name = self.name_layer + "_layernorm")
        self.activation = layers.Activation('gelu', name = self.name_layer + "_activation")

    def get_config(self):

        return {'name': self.name_layer, 'pool_list': self.pool_list}

    def call(self, x, mask=None):

        input_shape = tf.shape(x)

        row_length = [tf.cast(input_shape[1], 'float32') / i for i in self.pool_list]
        col_length = [tf.cast(input_shape[2], 'float32') / i for i in self.pool_list]

        outputs = []

        for pool_num, num_pool_regions in enumerate(self.pool_list):

            for jy in range(num_pool_regions):

                for ix in range(num_pool_regions):

                    x1 = ix * col_length[pool_num]
                    x2 = ix * col_length[pool_num] + col_length[pool_num]
                    y1 = jy * row_length[pool_num]
                    y2 = jy * row_length[pool_num] + row_length[pool_num]

                    x1 = tf.cast(tf.round(x1), dtype = tf.int32)
                    x2 = tf.cast(tf.round(x2), dtype = tf.int32)
                    y1 = tf.cast(tf.round(y1), dtype = tf.int32)
                    y2 = tf.cast(tf.round(y2), dtype = tf.int32)

                    new_shape = [input_shape[0], y2 - y1, x2 - x1, input_shape[3]]

                    x_crop = x[:, y1:y2, x1:x2, :]
                    xm = tf.reshape(tensor = x_crop, shape = new_shape)
                    pooled_val = tf.reduce_max(xm, axis=(1, 2))
                    outputs.append(pooled_val)

        outputs = tf.concat(outputs, axis = 1)

        return self.activation(self.norm(outputs))