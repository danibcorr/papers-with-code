# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras import layers, models

# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# --------------------------------------------------------------------------------------------

@tf.keras.saving.register_keras_serializable(package='Double_Residual_Block')
class Double_Residual_Block(layers.Layer):

    """
    Residual block with two convolutional layers.

    Args:
        num_filters (int): Number of filters in the convolutional layers.
        stride (int): Stride for the convolutional layers.
        downsampling (bool): Whether to apply downsampling.
    """

    def __init__(self, num_filters: int, stride: int, downsampling: bool):

        """
        Initializes the Residual Block with two convolutional layers.

        Args:
            num_filters (int): Number of filters in the convolutional layers.
            stride (int): Stride for the convolutional layers.
            downsampling (bool): Whether to apply downsampling.
        """

        super(Double_Residual_Block, self).__init__()

        self.downsampling = downsampling
        self.num_filters = num_filters
        self.stride = stride

        self.branch_block1 = models.Sequential([
            layers.Conv2D(filters=num_filters, kernel_size=3, strides=stride, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(filters=num_filters, kernel_size=3, strides=1, padding='same'),
            layers.BatchNormalization()
        ])

    def get_config(self) -> dict:

        """
        Returns the configuration of the layer.
        
        Returns:
            dict: Configuration of the layer.
        """

        return {'num_filters': self.num_filters, 'stride': self.stride, 'downsampling': self.downsampling}

    def call(self, input_tensor: tf.Tensor) -> tf.Tensor:

        """
        Forward pass of the Residual Block.

        Args:
            input_tensor (tf.Tensor): Input tensor to the block.

        Returns:
            tf.Tensor: Output tensor after applying the residual block.
        """

        if self.downsampling:

            branch_connection_input2 = layers.Conv2D(filters=self.num_filters, kernel_size=1, strides=self.stride, padding='same')(input_tensor)
            branch_connection_output2 = layers.BatchNormalization()(branch_connection_input2)

        else:

            branch_connection_output2 = input_tensor

        branch_connection_output1 = self.branch_block1(input_tensor)
        concatenation = layers.Add()([branch_connection_output1, branch_connection_output2])
        
        return layers.ReLU()(concatenation)

@tf.keras.saving.register_keras_serializable(package='Triple_Residual_Block')
class Triple_Residual_Block(layers.Layer):

    """
    Residual block with three convolutional layers.

    Args:
        num_filters (int): Number of filters in the convolutional layers.
        increase (int): Multiplicative factor for the number of filters in the final convolution.
        stride (int): Stride for the convolutional layers.
    """

    def __init__(self, num_filters: int, increase: int, stride: int):

        """
        Initializes the Residual Block with three convolutional layers.

        Args:
            num_filters (int): Number of filters in the convolutional layers.
            increase (int): Multiplicative factor for the number of filters in the final convolution.
            stride (int): Stride for the convolutional layers.
        """

        super(Triple_Residual_Block, self).__init__()

        self.num_filters = num_filters
        self.increase = increase
        self.stride = stride

        self.branch_block1 = models.Sequential([
            layers.Conv2D(filters=num_filters, kernel_size=1, strides=stride, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(filters=num_filters, kernel_size=3, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(filters=num_filters * increase, kernel_size=1, strides=1, padding='same'),
            layers.BatchNormalization()
        ])

    def get_config(self) -> dict:

        """
        Returns the configuration of the layer.
        
        Returns:
            dict: Configuration of the layer.
        """

        return {'num_filters': self.num_filters, 'increase': self.increase, 'stride': self.stride}

    def call(self, input_tensor: tf.Tensor) -> tf.Tensor:

        """
        Forward pass of the Residual Block with three convolutional layers.

        Args:
            input_tensor (tf.Tensor): Input tensor to the block.

        Returns:
            tf.Tensor: Output tensor after applying the residual block.
        """

        branch_connection_input2 = layers.Conv2D(filters=self.num_filters * self.increase, kernel_size=1, strides=self.stride, padding='same')(input_tensor)
        branch_connection_output2 = layers.BatchNormalization()(branch_connection_input2)

        branch_connection_output1 = self.branch_block1(input_tensor)
        concatenation = layers.Add()([branch_connection_output1, branch_connection_output2])
        
        return layers.ReLU()(concatenation)

@tf.keras.saving.register_keras_serializable(package='ResNet')
class ResNet(tf.keras.Model):

    """
    ResNet architecture with variable configurations.

    Args:
        configuration (list): List defining the number of residual blocks for different layers.
        increase (int): Multiplicative factor for the number of filters in certain blocks.
        num_clases (int): Number of output classes.
    """

    def __init__(self, configuration: list, increase: int, num_clases: int):

        """
        Initializes the ResNet model with specified configurations.

        Args:
            configuration (list): List defining the number of residual blocks for different layers.
            increase (int): Multiplicative factor for the number of filters in certain blocks.
            num_clases (int): Number of output classes.
        """

        super(ResNet, self).__init__()

        self.configuration = configuration
        self.increase = increase
        self.num_clases = num_clases

        self.architecture = models.Sequential([
            layers.Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=7, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        ])

        if configuration[0] in [18, 34]:

            for i in range(configuration[1]):

                self.architecture.add(Double_Residual_Block(num_filters=64, stride=1, downsampling=False))

            for i in range(configuration[2]):

                if i == 0:

                    self.architecture.add(Double_Residual_Block(num_filters=128, stride=2, downsampling=True))

                else:

                    self.architecture.add(Double_Residual_Block(num_filters=128, stride=1, downsampling=False))

            for i in range(configuration[3]):

                if i == 0:

                    self.architecture.add(Double_Residual_Block(num_filters=256, stride=2, downsampling=True))

                else:

                    self.architecture.add(Double_Residual_Block(num_filters=256, stride=1, downsampling=False))

            for i in range(configuration[4]):

                if i == 0:

                    self.architecture.add(Double_Residual_Block(num_filters=512, stride=2, downsampling=True))

                else:

                    self.architecture.add(Double_Residual_Block(num_filters=512, stride=1, downsampling=False))

        if configuration[0] in [50, 101, 152]:

            for i in range(configuration[1]):

                self.architecture.add(Triple_Residual_Block(num_filters=64, increase=increase, stride=1))

            for i in range(configuration[2]):

                if i == 0:

                    self.architecture.add(Triple_Residual_Block(num_filters=128, increase=increase, stride=2))

                else:

                    self.architecture.add(Triple_Residual_Block(num_filters=128, increase=increase, stride=1))

            for i in range(configuration[3]):

                if i == 0:

                    self.architecture.add(Triple_Residual_Block(num_filters=256, increase=increase, stride=2))

                else:

                    self.architecture.add(Triple_Residual_Block(num_filters=256, increase=increase, stride=1))

            for i in range(configuration[4]):

                if i == 0:

                    self.architecture.add(Triple_Residual_Block(num_filters=512, increase=increase, stride=2))

                else:

                    self.architecture.add(Triple_Residual_Block(num_filters=512, increase=increase, stride=1))

        self.architecture.add(layers.GlobalAveragePooling2D())
        self.architecture.add(layers.Flatten())
        self.architecture.add(layers.Dense(units=num_clases, activation='softmax'))

    def get_config(self) -> dict:

        """
        Returns the configuration of the ResNet model.
        
        Returns:
            dict: Configuration of the model.
        """

        return {'configuration': self.configuration, 'increase': self.increase, 'num_clases': self.num_clases}

    def call(self, inputs: tf.Tensor) -> tf.Tensor:

        """
        Forward pass of the GoogLeNet model.
        
        Args:
            inputs (tf.Tensor): Input tensor to the GoogLeNet model.
        
        Returns:
            tf.Tensor: Output tensor after applying the network.
        """
        
        return self.architecture(inputs)

if __name__ == '__main__':

    model_name = 'resnet152'

    x = tf.random.normal(shape=(2, 224, 224, 3))
    x = tf.convert_to_tensor(x)

    if model_name == 'resnet18':

        model = ResNet(configuration=[18, 2, 2, 2, 2], increase=1, num_clases=1000)

    elif model_name == 'resnet34':

        model = ResNet(configuration=[34, 3, 4, 6, 3], increase=1, num_clases=1000)

    elif model_name == 'resnet50':

        model = ResNet(configuration=[50, 3, 4, 6, 3], increase=4, num_clases=1000)

    elif model_name == 'resnet101':

        model = ResNet(configuration=[101, 3, 4, 23, 3], increase=4, num_clases=1000)

    elif model_name == 'resnet152':

        model = ResNet(configuration=[152, 3, 8, 36, 3], increase=4, num_clases=1000)

    model.summary()