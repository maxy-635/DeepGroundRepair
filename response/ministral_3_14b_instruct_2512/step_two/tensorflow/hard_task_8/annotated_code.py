import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Concatenate, Reshape, Lambda, Dense, Permute
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()
        self.num_classes = num_classes

        # Block 1 - Primary path
        self.conv1x1_1 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')
        self.depthwise_conv_1 = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1, activation='relu')
        self.conv1x1_2 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')

        # Block 1 - Branch path
        self.depthwise_conv_2 = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1, activation='relu')
        self.conv1x1_3 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')

        # Block 2 - Channel shuffling
        self.reshape = Reshape(target_shape=(None, None, 4, -1))  # Reshape to (height, width, groups, channels_per_group)
        self.permute = Permute(dims=(1, 2, 4, 3))  # Swap third and fourth dimensions
        self.reshape_back = Reshape(target_shape=(None, None, -1))  # Reshape back to original shape

        # Final classification layer
        self.flatten = tf.keras.layers.Flatten()
        self.dense = Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        # Block 1 - Primary path
        x1 = self.conv1x1_1(inputs)
        x1 = self.depthwise_conv_1(x1)
        x1 = self.conv1x1_2(x1)

        # Block 1 - Branch path
        x2 = self.depthwise_conv_2(inputs)
        x2 = self.conv1x1_3(x2)

        # Concatenate along channel dimension
        x = Concatenate(axis=-1)([x1, x2])

        # Block 2 - Channel shuffling
        x = self.reshape(x)
        x = self.permute(x)
        x = self.reshape_back(x)

        # Final classification
        x = self.flatten(x)
        outputs = self.dense(x)

        return outputs

if __name__ == "__main__":
    model = DL_Model()
    # MNIST images are 28x28 with 1 channel
    input_tensor = tf.keras.Input(shape=(28, 28, 1))
    output_tensor = model(input_tensor)
    print("Model summary:")
    model.summary()