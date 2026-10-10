import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Add, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # 1x1 convolution to increase dimensionality
        self.conv1x1_increase = Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )

        # 3x3 depthwise separable convolution for feature extraction
        self.depthwise_conv = DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            depth_multiplier=1,
            activation='relu'
        )

        # 1x1 convolution to reduce dimensionality
        self.conv1x1_reduce = Conv2D(
            filters=32,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )

        # Add layer for skip connection
        self.add = Add()

        # Flatten and dense layers
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Original input for skip connection
        x = inputs

        # First 1x1 conv to increase dimensionality
        x = self.conv1x1_increase(x)

        # Depthwise separable conv
        x = self.depthwise_conv(x)

        # Second 1x1 conv to reduce dimensionality
        x = self.conv1x1_reduce(x)

        # Add skip connection
        x = self.add([x, inputs])

        # Flatten and dense layers
        x = self.flatten(x)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # MNIST images are 28x28 with 1 channel
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)