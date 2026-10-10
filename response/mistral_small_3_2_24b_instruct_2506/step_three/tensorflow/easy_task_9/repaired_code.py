import tensorflow as tf
from tensorflow.keras.layers import Layer

class DL_Model(Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # 1x1 convolutional layer to increase dimensionality
        self.conv1x1_expand = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation=None
        )
        # 3x3 depthwise separable convolutional layer for feature extraction
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            depth_multiplier=1,
            activation=None
        )
        # 1x1 convolutional layer to reduce dimensionality
        self.conv1x1_reduce = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation=None
        )

    def call(self, inputs):
        # Store the original input for residual connection
        original_input = inputs

        # Apply 1x1 convolution to increase dimensionality
        x = self.conv1x1_expand(inputs)

        # Apply depthwise convolution for feature extraction
        x = self.depthwise_conv(x)

        # Apply 1x1 convolution to reduce dimensionality
        x = self.conv1x1_reduce(x)

        # Add the original input to the processed output (residual connection)
        x = tf.math.add(x, original_input)

        # Flatten the output
        x = tf.keras.layers.Flatten()(x)

        # Apply fully connected layer for classification
        x = tf.keras.layers.Dense(units=10, activation='softmax')(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # MNIST image shape
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)