import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, DepthwiseConv2D, Dense, Reshape, Permute
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Primary path
        self.conv1x1_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')
        self.depthwise_conv3x3 = DepthwiseConv2D(kernel_size=(3, 3), padding='same')
        self.conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')

        # Block 1: Branch path
        self.depthwise_conv3x3_branch = DepthwiseConv2D(kernel_size=(3, 3), padding='same')
        self.conv1x1_branch = Conv2D(filters=64, kernel_size=(1, 1), padding='same')

        # Block 2: Channel shuffling
        self.reshape1 = Reshape(target_shape=(-1, 28, 28, 4, 16))  # Assuming groups=4, channels_per_group=16
        self.permute = Permute(dims='mask')
        self.reshape2 = Reshape(target_shape=(-1, 28, 28, 64))

        # Final classification layer
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        # Block 1: Primary path
        x = self.conv1x1_1(inputs)  # output tensor shape: (1, 28, 28, 64)
        x = self.depthwise_conv3x3(x)  # output tensor shape: (1, 28, 28, 64)
        x = self.conv1x1_2(x)  # output tensor shape: (1, 28, 28, 64)

        # Block 1: Branch path
        branch = self.depthwise_conv3x3_branch(inputs)  # output tensor shape: (1, 28, 28, 1)
        branch = self.conv1x1_branch(branch)  # output tensor shape: (1, 28, 28, 64)

        # Concatenate along channel dimension
        x = tf.experimental.numpy.concatenate([x, branch], axis=-1)  # output tensor shape: (1, 28, 28, 128)

        # Block 2: Channel shuffling
        x = self.reshape1(x)  # output tensor shape: (1, 2, 28, 28, 4, 16)
        x = self.permute(x)
        x = self.reshape2(x)

        # Global average pooling and classification
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)