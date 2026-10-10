import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1_group1 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1_group2 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1_group3 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Block 2
        self.block2 = tf.keras.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu'),
        ])

        # Block 3
        self.depthwise_separable = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch
        self.branch_conv = Conv2D(filters='mask', kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Fully Connected Layer
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Block 1
        x = tf.split(inputs, num_or_size_splits=3, axis=-1)  # output tensor shape: (unknown)
        x1 = self.conv1_group1(x[0])  # output tensor shape: (1, 32, 32, 1)
        x2 = self.conv1_group2(x[1])  # output tensor shape: (1, 32, 32, 1)
        x3 = self.conv1_group3(x[2])  # output tensor shape: (1, 32, 32, 1)
        x = tf.concat([x1, x2, x3], axis=-1)  # output tensor shape: (1, 32, 32, 3)

        # Block 2
        shape = tf.shape(x)  # output tensor shape: (4)
        height = shape[1]  # output tensor shape: (1)
        width = shape[2]  # output tensor shape: (1)
        channels = shape[3]  # output tensor shape: (1)
        groups = 3  # output tensor shape: (1)
        channels_per_group = channels // groups  # output tensor shape: (1)
        x = tf.reshape(x, (shape[0], height, width, groups, channels_per_group))  # output tensor shape: (1, 32, 32, 3, 1)
        x = tf.transpose(x, perm=[0, 1, 2, 4, 3])  # output tensor shape: (1, 32, 32, 1, 3)
        x = tf.reshape(x, (shape[0], height, width, channels))  # output tensor shape: (1, 32, 32, 3)

        # Block 3
        x = self.block2(x)  # output tensor shape: (1, 32, 32, 3)

        # Branch
        branch = self.branch_conv(inputs)  # output tensor shape of this layer should be (1, 32, 32, 3)

        # Combine main path and branch
        x = tf.math.add(x, branch)

        # Fully Connected Layer
        x = tf.keras.layers.Flatten()(x)
        x = self.dense(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)