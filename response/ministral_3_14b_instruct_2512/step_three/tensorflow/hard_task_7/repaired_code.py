import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, Flatten, Concatenate, Lambda, Reshape, Permute
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolutional layer
        self.initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Block 1 components
        self.conv1x1_1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.conv1x1_2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Block 2 components
        self.reshape = Reshape(target_shape=(None, None, 4, 16))
        self.permute = Permute(dims=(1, 2, 4, 3))
        self.reshape_back = Reshape(target_shape=None)

        # Final classification layer
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Initial convolution
        x = self.initial_conv(inputs)

        # Block 1
        # Split the input into two groups along the last dimension
        split = tf.split(x, num_or_size_splits=2, axis=-1)
        x1 = split[0]
        x2 = split[1]

        # Process first group
        x1 = self.conv1x1_1(x1)
        x1 = self.depthwise_conv(x1)
        x1 = self.conv1x1_2(x1)

        # Concatenate both groups
        x = Concatenate(axis=-1)([x1, x2])

        # Block 2 - Channel shuffling
        # Get input shape
        input_shape = tf.shape(x)
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]

        # Reshape to (height, width, groups, channels_per_group)
        groups = 4
        channels_per_group = channels // groups
        x = Reshape((height, width, groups, channels_per_group))(x)

        # Permute dimensions
        x = self.permute(x)

        # Reshape back to original shape
        x = self.reshape_back(x)

        # Final classification
        x = self.flatten(x)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # MNIST images are 28x28 with 1 channel
    input_tensor = tf.random.normal(shape=(32, 28, 28, 1))  # batch_size=32
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)