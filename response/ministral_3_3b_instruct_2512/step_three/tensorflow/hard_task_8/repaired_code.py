import tensorflow as tf
from tensorflow.keras.layers import Conv1D, DepthwiseConv2D, Conv2D, Reshape, Permute, Flatten, Dense, Concatenate

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Primary path
        self.conv1_1 = Conv1D(filters=32, kernel_size=1, strides=1, padding='valid', activation='relu')
        self.depthwise_conv1 = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1, activation='relu')
        self.conv1_2 = Conv1D(filters=64, kernel_size=1, strides=1, padding='valid', activation='relu')

        # Branch path
        self.depthwise_conv2 = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1, activation='relu')
        self.conv2_1 = Conv1D(filters=64, kernel_size=1, strides=1, padding='valid', activation='relu')

        # Concatenate the outputs
        self.concatenate = Concatenate(axis=-1)

        # Block 2: Channel Shuffling
        self.reshape = Reshape(target_shape=(-1, 2, 16, 64))
        self.permute = Permute(dims=(1, 3, 2))
        self.reshape2 = Reshape(target_shape=(-1, 16, 64))

        # Fully connected layer
        self.flatten = Flatten()
        self.dense = Dense(10, activation='softmax')

    def call(self, inputs):
        # Block 1: Primary path
        x1 = self.conv1_1(inputs)
        x1 = self.depthwise_conv1(x1)
        x1 = self.conv1_2(x1)

        # Branch path
        x2 = self.depthwise_conv2(inputs)
        x2 = self.conv2_1(x2)

        # Concatenate the outputs
        x = self.concatenate([x1, x2])

        # Block 2: Channel Shuffling
        x = self.reshape(x)
        x = self.permute(x)
        x = self.reshape2(x)

        # Fully connected layer
        x = self.flatten(x)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28))  # Example input tensor with shape (batch_size, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)