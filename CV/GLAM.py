# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .CBAM import GlobalMinPooling2D, ChannelAttentionModule, SpatialAttentionModule

# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# https://arxiv.org/abs/2107.08000
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

class GlobalChannelAttention(layers.Layer):

    def __init__(self, in_channels, kernel_size):

        super(GlobalChannelAttention, self).__init__()

        assert (kernel_size % 2 == 1), "Kernel size must be odd"
        
        self.conv_q = layers.Conv1D(in_channels, kernel_size, 1, padding = 'same', activation = 'sigmoid')
        self.conv_k = layers.Conv1D(in_channels, kernel_size, 1, padding = 'same', activation = 'sigmoid')
        self.GAP = layers.GlobalAveragePooling2D(keepdims = True)
        
    def call(self, x):

        input_shape = tf.shape(x)
        N, H, W, C = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        
        # Global average pooling for channel-wise attention
        gap = tf.reshape(self.GAP(x), [N, 1, C])

        # Efficiently calculate query and key using convolutions
        query = self.conv_q(gap)
        key = self.conv_k(gap)

        # Calculate attention weights using matrix multiplication
        query_key = tf.matmul(key, query, transpose_a = True)
        query_key = tf.nn.softmax(query_key, axis = -1)  

        # Reshape to match feature map dimensions
        value = tf.reshape(x, [N, H * W, C])
        att = tf.matmul(value, query_key)
        att = tf.reshape(att, [N, H, W, C])

        return x * att

class GlobalSpatialAttention(layers.Layer):

    def __init__(self, in_channels, num_reduced_channels):

        super(GlobalSpatialAttention, self).__init__()
        
        self.conv1x1_q = layers.SeparableConv2D(num_reduced_channels, 1, 1)
        self.conv1x1_k = layers.SeparableConv2D(num_reduced_channels, 1, 1)
        self.conv1x1_v = layers.SeparableConv2D(num_reduced_channels, 1, 1)
        self.conv1x1_att = layers.SeparableConv2D(in_channels, 1, 1)

    def call(self, feature_maps, global_channel_output):

        query = self.conv1x1_q(feature_maps)
        key = self.conv1x1_k(feature_maps)
        value = self.conv1x1_v(feature_maps)

        # Reshape for compatibility with TensorFlow's batch-first format
        query_shape = tf.shape(query)
        N, H, W, C = query_shape[0], query_shape[1], query_shape[2], query_shape[3]

        query = tf.reshape(query, (N, H * W, C))
        key = tf.reshape(key, (N, H * W, C))
        value = tf.reshape(value, (N, H * W, C))

        # Efficiently calculate attention weights using matrix multiplication
        query_key = tf.matmul(key, query, transpose_b = True)  # Perform matmul in TensorFlow
        query_key = tf.nn.softmax(query_key, axis=-1)

        # Attention-weighted sum using a single matrix multiplication
        att = tf.matmul(value, query_key, transpose_a = True)  # Use matmul for efficiency
        att = tf.reshape(att, (N, H, W, C))
        att = self.conv1x1_att(att)
        
        return (global_channel_output * att) + global_channel_output

class LocalChannelAttention(layers.Layer):

    def __init__(self, in_channels, kernel_size):

        super(LocalChannelAttention, self).__init__()

        assert (kernel_size%2 == 1), "Kernel size must be odd"
        
        self.conv = layers.Conv1D(in_channels, kernel_size, 1, padding = 'same', activation = 'sigmoid')
        self.GAP = layers.GlobalAveragePooling2D(keepdims = True)
    
    def call(self, x):

        input_shape = tf.shape(x)
        N, H, W, C = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

        att = self.GAP(x)
        att = self.conv(att)

        return (x * att) + x
    
class LocalSpatialAttention(layers.Layer):

    def __init__(self, in_channels, num_reduced_channels):

        super(LocalSpatialAttention, self).__init__()
        
        self.conv1x1_1 = layers.SeparableConv2D(num_reduced_channels, 1, 1)
        self.conv1x1_2 = layers.SeparableConv2D(1, 1, 1)
        
        self.conv3x3 = layers.SeparableConv2D(num_reduced_channels, 3, 1, padding = 'same', dilation_rate = 1)
        self.conv5x5 = layers.SeparableConv2D(num_reduced_channels, 5, 1, padding = 'same', dilation_rate = 3)
        self.conv7x7 = layers.SeparableConv2D(num_reduced_channels, 7, 1, padding = 'same', dilation_rate = 5)
        
    def call(self, feature_maps, local_channel_output):

        att = self.conv1x1_1(feature_maps)

        d1 = self.conv3x3(att)
        d2 = self.conv5x5(att)
        d3 = self.conv7x7(att)

        att = tf.concat([att, d1, d2, d3], axis = -1)
        att = self.conv1x1_2(att)

        return (local_channel_output * att) + local_channel_output

class GLAM(layers.Layer):
    
    def __init__(self, in_channels, num_reduced_channels, kernel_size, name, use_cbam_local_attention = False):
        
        super(GLAM, self).__init__()
        
        self.use_cbam_local_attention = use_cbam_local_attention
        
        if use_cbam_local_attention == False:
		
            # GLAM Local Attention Implementation
            self.local_channel_att = LocalChannelAttention(in_channels, kernel_size)
            self.local_spatial_att = LocalSpatialAttention(in_channels, num_reduced_channels)
        
        else:
            
            # CBAM Local Attention Implementation
            self.local_channel_att = ChannelAttentionModule("GLAM_" + name)
            self.local_spatial_att = SpatialAttentionModule("GLAM_" + name)

        # GLAM Global Attention Implementation
        self.global_channel_att = GlobalChannelAttention(in_channels, kernel_size)
        self.global_spatial_att = GlobalSpatialAttention(in_channels, num_reduced_channels)
        
        self.fusion_weights = tf.Variable([0.333, 0.333, 0.333], trainable = True)
        
    def call(self, x):

        local_channel_att = self.local_channel_att(x)     
        
        if use_cbam_local_attention == False:
        
            # GLAM Implementation              
            local_att = self.local_spatial_att(x, local_channel_att)  
        
        else:
        
            # CBAM Implementation       
            local_att = (local_channel_att * self.local_spatial_att(x)) + local_channel_att  

        global_channel_att = self.global_channel_att(x)                 
        global_att = self.global_spatial_att(x, global_channel_att)
        
        local_att = tf.expand_dims(local_att, axis = 1) 
        global_att = tf.expand_dims(global_att, axis = 1)
        x = tf.expand_dims(x, axis = 1) 
        
        all_feature_maps = tf.concat([local_att, x, global_att], axis = 1)
        weights = tf.reshape(tf.nn.softmax(self.fusion_weights, axis = -1), (1, 3, 1, 1, 1))
        fused_feature_maps = tf.reduce_sum(all_feature_maps * weights, axis = 1)
        
        return fused_feature_maps
