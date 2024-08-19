# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .APS import CircularPad
from .CBAM import GlobalMinPooling2D, ChannelAttentionModule, SpatialAttentionModule

# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# https://arxiv.org/abs/2107.08000
# --------------------------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package='GlobalChannelAttention')
class GlobalChannelAttention(layers.Layer):

    """
    Global channel attention module.

    Args:
        in_channels (int): Number of input channels.
        kernel_size (int): Size of the kernel for the convolutional layer.
    """

    def __init__(self, in_channels: int, kernel_size: int):

        super(GlobalChannelAttention, self).__init__()

        assert (kernel_size % 2 == 0), "Kernel size must be even."
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(filters=1, kernel_size=kernel_size, padding="same")
    
    def get_config(self) -> dict:
        
        """
        Returns the configuration of the layer.
        
        Returns:
            dict: Configuration of the layer.
        """

        return {'in_channels': self.in_channels, 'kernel_size': self.kernel_size}

    def call(self, x: tf.Tensor) -> tf.Tensor:

        """
        Forward pass of the global channel attention module.

        Args:
            x (tf.Tensor): Input tensor.
            
        Returns:
            tf.Tensor: Output tensor after applying global channel attention.
        """

        input_shape = tf.shape(x)
        N, H, W, C = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        att = self.conv(tf.reduce_mean(x, axis=-1))

        return x * att + x

@keras.saving.register_keras_serializable(package='LocalChannelAttention')
class LocalChannelAttention(layers.Layer):

    """
    Local channel attention module.

    Args:
        in_channels (int): Number of input channels.
        kernel_size (int): Size of the kernel for the convolutional layer.
    """

    def __init__(self, in_channels: int, kernel_size: int):

        super(LocalChannelAttention, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(filters=1, kernel_size=kernel_size, padding="same")
    
    def get_config(self) -> dict:
        
        """
        Returns the configuration of the layer.
        
        Returns:
            dict: Configuration of the layer.
        """

        return {'in_channels': self.in_channels, 'kernel_size': self.kernel_size}

    def call(self, x: tf.Tensor) -> tf.Tensor:

        """
        Forward pass of the local channel attention module.

        Args:
            x (tf.Tensor): Input tensor.
            
        Returns:
            tf.Tensor: Output tensor after applying local channel attention.
        """

        input_shape = tf.shape(x)
        N, H, W, C = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        att = self.conv(tf.reduce_mean(x, axis=-1))

        return x * att + x

@keras.saving.register_keras_serializable(package='GlobalSpatialAttention')
class GlobalSpatialAttention(layers.Layer):

    """
    Global spatial attention module.

    Args:
        in_channels (int): Number of input channels.
        num_reduced_channels (int): Number of reduced channels for the convolutional layer.
    """

    def __init__(self, in_channels: int, num_reduced_channels: int):

        super(GlobalSpatialAttention, self).__init__()
        
        self.in_channels = in_channels
        self.num_reduced_channels = num_reduced_channels
        self.conv = layers.Conv2D(filters=num_reduced_channels, kernel_size=1, padding="same")
        
    def get_config(self) -> dict:
        
        """
        Returns the configuration of the layer.
        
        Returns:
            dict: Configuration of the layer.
        """

        return {'in_channels': self.in_channels, 'num_reduced_channels': self.num_reduced_channels}

    def call(self, x: tf.Tensor) -> tf.Tensor:

        """
        Forward pass of the global spatial attention module.

        Args:
            x (tf.Tensor): Input tensor.
            
        Returns:
            tf.Tensor: Output tensor after applying global spatial attention.
        """

        input_shape = tf.shape(x)
        N, H, W, C = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        att = self.conv(tf.reduce_mean(x, axis=-1))

        return x * att + x

@keras.saving.register_keras_serializable(package='LocalSpatialAttention')
class LocalSpatialAttention(layers.Layer):

    """
    Local spatial attention module.

    Args:
        in_channels (int): Number of input channels.
        num_reduced_channels (int): Number of reduced channels for the convolutional layer.
    """

    def __init__(self, in_channels: int, num_reduced_channels: int):

        super(LocalSpatialAttention, self).__init__()
        
        self.in_channels = in_channels
        self.num_reduced_channels = num_reduced_channels
        self.conv_3x3 = layers.Conv2D(filters=num_reduced_channels, kernel_size=3, padding="same", dilation_rate=1)
        self.conv_5x5 = layers.Conv2D(filters=num_reduced_channels, kernel_size=5, padding="same", dilation_rate=3)
        self.conv_7x7 = layers.Conv2D(filters=num_reduced_channels, kernel_size=7, padding="same", dilation_rate=5)
        
    def get_config(self) -> dict:
        
        """
        Returns the configuration of the layer.
        
        Returns:
            dict: Configuration of the layer.
        """

        return {'in_channels': self.in_channels, 'num_reduced_channels': self.num_reduced_channels}

    def call(self, x: tf.Tensor) -> tf.Tensor:

        """
        Forward pass of the local spatial attention module.

        Args:
            x (tf.Tensor): Input tensor.
            
        Returns:
            tf.Tensor: Output tensor after applying local spatial attention.
        """

        att = self.conv_3x3(x) + self.conv_5x5(x) + self.conv_7x7(x)
        
        return x * att + x

@keras.saving.register_keras_serializable(package='GLAM')
class GLAM(layers.Layer):

    """
    Global-local attention module.

    Args:
        in_channels (int): Number of input channels.
        num_reduced_channels (int): Number of reduced channels for the convolutional layer.
        kernel_size (int): Size of the kernel for the convolutional layer.
        name (str): Name of the GLAM instance.
        use_cbam_local_attention (bool): Whether to use CBAM local attention. Defaults to False.
    """

    def __init__(self, in_channels: int, num_reduced_channels: int, kernel_size: int, name: str = "GLAM", use_cbam_local_attention: bool = False):

        super(GLAM, self).__init__(name=name)
        
        self.in_channels = in_channels
        self.num_reduced_channels = num_reduced_channels
        self.kernel_size = kernel_size
        self.name = name
        self.use_cbam_local_attention = use_cbam_local_attention

        if use_cbam_local_attention:

            self.local_channel_att = LocalChannelAttention(in_channels=in_channels, kernel_size=kernel_size)
            self.local_spatial_att = LocalSpatialAttention(in_channels=in_channels, num_reduced_channels=num_reduced_channels)

        else:

            self.global_channel_att = GlobalChannelAttention(in_channels=in_channels, kernel_size=kernel_size)
            self.global_spatial_att = GlobalSpatialAttention(in_channels=in_channels, num_reduced_channels=num_reduced_channels)
            
    def get_config(self) -> dict:
        
        """
        Returns the configuration of the layer.
        
        Returns:
            dict: Configuration of the layer.
        """

        return {'in_channels': self.in_channels, 'num_reduced_channels': self.num_reduced_channels,
                'kernel_size': self.kernel_size, 'name': self.name, 'use_cbam_local_attention': self.use_cbam_local_attention}

    def call(self, x: tf.Tensor) -> tf.Tensor:

        """
        Forward pass of the GLAM module.

        Args:
            x (tf.Tensor): Input tensor.
            
        Returns:
            tf.Tensor: Output tensor after applying global-local attention.
        """
        
        if not self.use_cbam_local_attention:

            return self.global_channel_att(x) + self.global_spatial_att(x)
        
        return self.local_channel_att(x) + self.local_spatial_att(x)