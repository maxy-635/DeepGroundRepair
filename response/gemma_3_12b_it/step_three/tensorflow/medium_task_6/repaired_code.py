import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense, Softmax, Input

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial Convolution
        self.conv_initial = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=None
        )
        self.bn_initial = BatchNormalization(axis=-1)
        self.relu_initial = ReLU()

        # Block 1
        self.conv_block1 = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=None
        )
        self.bn_block1 = BatchNormalization(axis=-1)
        self.relu_block1 = ReLU()

        # Block 2
        self.conv_block2 = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=None
        )
        self.bn_block2 = BatchNormalization(axis=-1)
        self.relu_block2 = ReLU()

        # Block 3
        self.conv_block3 = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=None
        )
        self.bn_block3 = BatchNormalization(axis=-1)
        self.relu_block3 = ReLU()

        # Fully Connected Layers
        self.flatten = Flatten()
        self.dense1 = Dense(
            units=128,
            activation=ReLU()
        )
        self.dense2 = Dense(
            units=10,
            activation=None
        )
        self.softmax = Softmax(axis=-1)

    def call(self, inputs):
        # Initial Convolution Block
        x = self.conv_initial(inputs)
        x = self.bn_initial(x)
        x = self.relu_initial(x)

        # Parallel Blocks
        block1 = self.conv_block1(inputs)
        block1 = self.bn_block1(block1)
        block1 = self.relu_block1(block1)

        block2 = self.conv_block2(inputs)
        block2 = self.bn_block2(block2)
        block2 = self.relu_block2(block2)

        block3 = self.conv_block3(inputs)
        block3 = self.bn_block3(block3)
        block3 = self.relu_block3(block3)

        # Add outputs of blocks and initial convolution
        x = Add()([x, block1, block2, block3])

        # Flatten and Fully Connected Layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor)