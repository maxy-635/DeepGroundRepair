import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense, Activation, concatenate, GlobalAveragePooling2D

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolution layer
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        self.bn1 = BatchNormalization(axis=-1)
        self.relu1 = ReLU()

        # First parallel block
        self.conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        self.bn2 = BatchNormalization(axis=-1)
        self.relu2 = ReLU()

        # Second parallel block
        self.conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        self.bn3 = BatchNormalization(axis=-1)
        self.relu3 = ReLU()

        # Third parallel block
        self.conv4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        self.bn4 = BatchNormalization(axis=-1)
        self.relu4 = ReLU()

        # Concatenate the outputs
        self.add_layer = Add()

        # Flatten and fully connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation=None)
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Initial convolution
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        # First parallel block
        block1 = self.conv2(x)
        block1 = self.bn2(block1)
        block1 = self.relu2(block1)

        # Second parallel block
        block2 = self.conv3(x)
        block2 = self.bn3(block2)
        block2 = self.relu3(block2)

        # Third parallel block
        block3 = self.conv4(x)
        block3 = self.bn4(block3)
        block3 = self.relu4(block3)

        # Add the outputs
        x = self.add_layer([x, block1, block2, block3])

        # Flatten and fully connected layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))  # CIFAR-10 image dimensions
    output_tensor = model(input_tensor)
    print(output_tensor.shape)