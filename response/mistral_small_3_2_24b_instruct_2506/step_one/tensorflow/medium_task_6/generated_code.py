import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Dense, Flatten
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolution
        self.initial_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same')

        # Three parallel blocks
        self.block1_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.block1_bn = BatchNormalization(axis=-1)
        self.block1_relu = tf.nn.relu

        self.block2_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.block2_bn = BatchNormalization(axis=-1)
        self.block2_relu = tf.nn.relu

        self.block3_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.block3_bn = BatchNormalization(axis=-1)
        self.block3_relu = tf.nn.relu

        # Flatten and fully connected layers
        self.flatten = Flatten()
        self.fc1 = Dense(units=128, activation='relu')
        self.fc2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Initial convolution
        x = self.initial_conv(inputs)

        # Parallel blocks
        block1 = self.block1_relu(self.block1_bn(self.block1_conv(x)))
        block2 = self.block1_relu(self.block2_bn(self.block2_conv(x)))
        block3 = self.block1_relu(self.block3_bn(self.block3_conv(x)))

        # Add outputs
        x = tf.math.add(tf.math.add(block1, block2), block3)
        x = tf.math.add(x, x)  # Add with initial convolution output

        # Flatten and fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)