import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Lambda, Dense, Flatten, Permute
from tensorflow.keras.models import Model
from tensorflow.experimental import numpy as tnp

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolutional layer
        self.initial_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')

        # Block 1 components
        self.conv1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')
        self.depthwise_conv = Lambda(lambda x: tf.nn.depthwise_conv2d(x, filter=tf.constant(0.1, shape=[3, 3, 32, 1]), strides=[1, 1, 1, 1], padding='SAME'))
        self.conv1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')
        self.concat = Lambda(lambda x: tnp.concatenate(x, axis=-1))

        # Block 2 components
        self.reshape1 = Lambda(function='mask')
        self.permute = Permute(dims=(1, 2, 4, 3))
        self.reshape2 = Lambda(lambda x: tf.raw_ops.Reshape(tensor=x, shape=[-1, tf.shape(x)[1], tf.shape(x)[2], 32]))

        # Final layers
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        # Initial convolution
        x = self.initial_conv(inputs)  # output tensor shape: (1, 28, 28, 32)

        # Block 1
        # Split into two groups
        split = tf.split(x, num_or_size_splits=2, axis=-1)  # output tensor shape: (unknown)
        group1 = split[0]  # output tensor shape: (1, 28, 28, 16)
        group2 = split[1]  # output tensor shape: (1, 28, 28, 16)

        # Process group1
        group1 = self.conv1x1_1(group1)  # output tensor shape: (1, 28, 28, 32)
        group1 = self.depthwise_conv(group1)  # output tensor shape: (1, 28, 28, 32)
        group1 = self.conv1x1_2(group1)  # output tensor shape: (1, 28, 28, 32)

        # Concatenate groups
        x = self.concat([group1, group2])  # output tensor shape: (1, 28, 28, 48)

        # Block 2
        # Reshape into groups
        x = self.reshape1(x)
        # Permute dimensions
        x = self.permute(x)
        # Reshape back
        x = self.reshape2(x)

        # Final layers
        x = self.flatten(x)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # MNIST image shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (1, 10)