import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense, Softmax
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolution
        self.conv_initial = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')

        # Three parallel blocks
        self.conv_block1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn_block1 = BatchNormalization(axis=-1)
        self.relu_block1 = ReLU()

        self.conv_block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn_block2 = BatchNormalization(axis=-1)
        self.relu_block2 = ReLU()

        self.conv_block3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn_block3 = BatchNormalization(axis=-1)
        self.relu_block3 = ReLU()

        # Add layer for combining outputs
        self.add = Add()

        # Fully connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Initial convolution
        x = self.conv_initial(inputs)

        # Three parallel blocks
        block1 = self.relu_block1(self.bn_block1(self.conv_block1(x)))
        block2 = self.relu_block2(self.bn_block2(self.conv_block2(x)))
        block3 = self.relu_block3(self.bn_block3(self.conv_block3(x)))

        # Add the outputs of the blocks with the initial convolution's output
        x = self.add([x, block1, block2, block3])

        # Fully connected layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((32, 32, 32, 3))  # Batch size 32, 32x32 images with 3 channels
    output_tensor = model(input_tensor)
    print(output_tensor.shape)