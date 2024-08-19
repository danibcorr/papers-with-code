# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# --------------------------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package='ConvBlock')
class ConvBlock(layers.Layer):

    """
    A convolutional block consisting of a Conv2D layer followed by BatchNormalization.
    
    Attributes:
        num_filters (int): Number of filters in the Conv2D layer.
        kernel_size (int): Size of the kernel in the Conv2D layer.
        stride (int): Stride length of the Conv2D layer.
    """

    def __init__(self, num_filters: int, kernel_size: int, stride: int):

        """
        Initializes the ConvBlock with specified number of filters, kernel size, and stride.
        
        Args:
            num_filters (int): Number of filters in the Conv2D layer.
            kernel_size (int): Size of the kernel in the Conv2D layer.
            stride (int): Stride length of the Conv2D layer.
        """

        super(ConvBlock, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv_layer = models.Sequential([
            layers.Conv2D(filters=num_filters, kernel_size=kernel_size, padding='same', strides=stride, activation='relu'),
            layers.BatchNormalization()
        ])
    
    def get_config(self) -> dict:

        """
        Returns the configuration of the ConvBlock layer.
        
        Returns:
            dict: Configuration of the ConvBlock layer.
        """

        return {'num_filters': self.num_filters, 'kernel_size': self.kernel_size, 'stride': self.stride}

    def call(self, input_tensor: tf.Tensor) -> tf.Tensor:

        """
        Forward pass of the ConvBlock.
        
        Args:
            input_tensor (tf.Tensor): Input tensor to the ConvBlock.
        
        Returns:
            tf.Tensor: Output tensor after applying the Conv2D and BatchNormalization layers.
        """

        return self.conv_layer(input_tensor)

@keras.saving.register_keras_serializable(package='InceptionBlock')
class InceptionBlock(layers.Layer):

    """
    An Inception block as described in the GoogLeNet architecture.
    
    Attributes:
        num_fil_r1_1 (int): Number of filters for 1x1 convolution in the first branch.
        num_fil_r2_1 (int): Number of filters for 1x1 convolution in the second branch.
        num_fil_r2_3 (int): Number of filters for 3x3 convolution in the second branch.
        num_fil_r3_1 (int): Number of filters for 1x1 convolution in the third branch.
        num_fil_r3_5 (int): Number of filters for 5x5 convolution in the third branch.
        num_fil_r4_1 (int): Number of filters for 1x1 convolution in the fourth branch.
    """

    def __init__(self, num_fil_r1_1: int, num_fil_r2_1: int, num_fil_r2_3: int, 
                 num_fil_r3_1: int, num_fil_r3_5: int, num_fil_r4_1: int):

        """
        Initializes the InceptionBlock with specified number of filters for each branch.
        
        Args:
            num_fil_r1_1 (int): Number of filters for 1x1 convolution in the first branch.
            num_fil_r2_1 (int): Number of filters for 1x1 convolution in the second branch.
            num_fil_r2_3 (int): Number of filters for 3x3 convolution in the second branch.
            num_fil_r3_1 (int): Number of filters for 1x1 convolution in the third branch.
            num_fil_r3_5 (int): Number of filters for 5x5 convolution in the third branch.
            num_fil_r4_1 (int): Number of filters for 1x1 convolution in the fourth branch.
        """

        super(InceptionBlock, self).__init__()

        self.num_fil_r1_1 = num_fil_r1_1
        self.num_fil_r2_1 = num_fil_r2_1
        self.num_fil_r2_3 = num_fil_r2_3
        self.num_fil_r3_1 = num_fil_r3_1
        self.num_fil_r3_5 = num_fil_r3_5
        self.num_fil_r4_1 = num_fil_r4_1

        self.branch_1 = ConvBlock(num_filters=num_fil_r1_1, kernel_size=1, stride=1)

        self.branch_2 = models.Sequential([
            ConvBlock(num_filters=num_fil_r2_1, kernel_size=1, stride=1),
            ConvBlock(num_filters=num_fil_r2_3, kernel_size=3, stride=1)
        ])

        self.branch_3 = models.Sequential([
            ConvBlock(num_filters=num_fil_r3_1, kernel_size=1, stride=1),
            ConvBlock(num_filters=num_fil_r3_5, kernel_size=5, stride=1)
        ])

        self.branch_4 = models.Sequential([
            layers.MaxPool2D(pool_size=3, strides=1, padding='same'),
            ConvBlock(num_filters=num_fil_r4_1, kernel_size=1, stride=1)
        ])

    def get_config(self) -> dict:

        """
        Returns the configuration of the InceptionBlock layer.
        
        Returns:
            dict: Configuration of the InceptionBlock layer.
        """

        return {'num_fil_r1_1': self.num_fil_r1_1, 'num_fil_r2_1': self.num_fil_r2_1, 'num_fil_r2_3': self.num_fil_r2_3,
                'num_fil_r3_1': self.num_fil_r3_1, 'num_fil_r3_5': self.num_fil_r3_5, 'num_fil_r4_1': self.num_fil_r4_1}

    def call(self, input_tensor: tf.Tensor) -> tf.Tensor:

        """
        Forward pass of the InceptionBlock.
        
        Args:
            input_tensor (tf.Tensor): Input tensor to the InceptionBlock.
        
        Returns:
            tf.Tensor: Concatenated output of all branches.
        """

        return tf.concat([
            self.branch_1(input_tensor),
            self.branch_2(input_tensor),
            self.branch_3(input_tensor),
            self.branch_4(input_tensor)
        ], axis=3)

@keras.saving.register_keras_serializable(package='GoogLeNet')
class GoogLeNet(tf.keras.Model):

    """
    Implementation of the GoogLeNet (Inception v1) architecture.
    
    Attributes:
        input_shape (tuple): Shape of the input tensor.
        num_classes (int): Number of output classes for classification.
    """

    def __init__(self, input_shape: tuple = (224, 224, 3), num_classes: int = 10):

        """
        Initializes the GoogLeNet model with specified input shape and number of classes.
        
        Args:
            input_shape (tuple): Shape of the input tensor.
            num_classes (int): Number of output classes for classification.
        """

        super(GoogLeNet, self).__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes

        self.architecture = models.Sequential([
            # Initial layers
            layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same', activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
            
            ConvBlock(num_filters=192, kernel_size=3, stride=1),
            layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
            
            # Inception blocks
            InceptionBlock(num_fil_r1_1=64, num_fil_r2_1=96, num_fil_r2_3=128, num_fil_r3_1=16, num_fil_r3_5=32, num_fil_r4_1=32),
            InceptionBlock(num_fil_r1_1=128, num_fil_r2_1=128, num_fil_r2_3=192, num_fil_r3_1=32, num_fil_r3_5=96, num_fil_r4_1=64),
            layers.MaxPool2D(pool_size=3, strides=2, padding='same'),

            InceptionBlock(num_fil_r1_1=192, num_fil_r2_1=96, num_fil_r2_3=208, num_fil_r3_1=16, num_fil_r3_5=48, num_fil_r4_1=64),
            InceptionBlock(num_fil_r1_1=160, num_fil_r2_1=112, num_fil_r2_3=224, num_fil_r3_1=24, num_fil_r3_5=64, num_fil_r4_1=64),
            InceptionBlock(num_fil_r1_1=128, num_fil_r2_1=128, num_fil_r2_3=256, num_fil_r3_1=24, num_fil_r3_5=64, num_fil_r4_1=64),
            InceptionBlock(num_fil_r1_1=112, num_fil_r2_1=144, num_fil_r2_3=288, num_fil_r3_1=32, num_fil_r3_5=64, num_fil_r4_1=64),
            InceptionBlock(num_fil_r1_1=256, num_fil_r2_1=160, num_fil_r2_3=320, num_fil_r3_1=32, num_fil_r3_5=128, num_fil_r4_1=128),
            layers.MaxPool2D(pool_size=3, strides=2, padding='same'),

            InceptionBlock(num_fil_r1_1=256, num_fil_r2_1=160, num_fil_r2_3=320, num_fil_r3_1=32, num_fil_r3_5=128, num_fil_r4_1=128),
            InceptionBlock(num_fil_r1_1=384, num_fil_r2_1=192, num_fil_r2_3=384, num_fil_r3_1=48, num_fil_r3_5=128, num_fil_r4_1=128),
            layers.AveragePooling2D(pool_size=7, strides=1),

            # Final layers
            layers.Flatten(),
            layers.Dropout(rate=0.4),
            layers.Dense(units=1000, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(units=num_classes, activation='sigmoid'),
        ])

    def get_config(self) -> dict:
        
        """
        Returns the configuration of the GoogLeNet model.
        
        Returns:
            dict: Configuration of the GoogLeNet model.
        """

        return {'input_shape': self.input_shape, 'num_classes': self.num_classes}

    def call(self, inputs: tf.Tensor) -> tf.Tensor:

        """
        Forward pass of the GoogLeNet model.
        
        Args:
            inputs (tf.Tensor): Input tensor to the GoogLeNet model.
        
        Returns:
            tf.Tensor: Output tensor after applying the network.
        """
        
        return self.architecture(inputs)
