import tensorflow as tf
from tensorflow.keras import layers, datasets, losses


@tf.keras.utils.register_keras_serializable(package="LeNet")
class LeNet(tf.keras.Model):
    """
    LeNet neural network architecture.

    Args:
        input_shape (tuple, optional): Input shape of the data. Defaults to (28, 28, 1).
        num_classes (int, optional): Number of classes in the classification problem. Defaults to 10.
    """

    def __init__(self, input_shape: tuple = (28, 28, 1), num_classes: int = 10):
        """
        Initializes the LeNet model.

        Args:
            input_shape (tuple, optional): Input shape of the data. Defaults to (28, 28, 1).
            num_classes (int, optional): Number of classes in the classification problem. Defaults to 10.
        """

        super(LeNet, self).__init__()

        # Define the LeNet architecture
        self.architecture = tf.keras.models.Sequential(
            [
                layers.Conv2D(
                    filters=6,
                    kernel_size=5,
                    padding="same",
                    activation="relu",
                    input_shape=input_shape,
                ),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=2, strides=2),
                layers.Conv2D(filters=16, kernel_size=5, activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=2, strides=2),
                layers.Flatten(),
                layers.Dense(units=120, activation="relu"),
                layers.Dense(units=84, activation="relu"),
                layers.Dense(units=num_classes, activation="softmax"),
            ]
        )

    def get_config(self) -> dict:
        """
        Returns the configuration of the LeNet model.

        Returns:
            dict: Configuration of the LeNet model.
        """

        return {"input_shape": self.input_shape, "num_classes": self.num_classes}

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass of the LeNet model.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor.
        """

        return self.architecture(inputs)
