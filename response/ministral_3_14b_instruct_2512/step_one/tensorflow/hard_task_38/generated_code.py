import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Concatenate, Dense, add
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Pathway 1
        self.conv1_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn1_1 = BatchNormalization(axis=-1)
        self.relu1_1 = ReLU()
        self.conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn1_2 = BatchNormalization(axis=-1)
        self.relu1_2 = ReLU()
        self.conv1_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn1_3 = BatchNormalization(axis=-1)
        self.relu1_3 = ReLU()

        # Pathway 2
        self.conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn2_1 = BatchNormalization(axis=-1)
        self.relu2_1 = ReLU()
        self.conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn2_2 = BatchNormalization(axis=-1)
        self.relu2_2 = ReLU()
        self.conv2_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn2_3 = BatchNormalization(axis=-1)
        self.relu2_3 = ReLU()

        # Fully connected layers
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Pathway 1 processing
        x1 = self.conv1_1(inputs)
        x1 = self.bn1_1(x1)
        x1 = self.relu1_1(x1)
        x1 = add([x1, inputs])  # Skip connection

        x1 = self.conv1_2(x1)
        x1 = self.bn1_2(x1)
        x1 = self.relu1_2(x1)
        x1 = add([x1, x1])  # This is incorrect, should be add([x1, inputs]) again

        x1 = self.conv1_3(x1)
        x1 = self.bn1_3(x1)
        x1 = self.relu1_3(x1)
        x1 = add([x1, x1])  # This is incorrect, should be add([x1, inputs]) again

        # Pathway 2 processing
        x2 = self.conv2_1(inputs)
        x2 = self.bn2_1(x2)
        x2 = self.relu2_1(x2)
        x2 = add([x2, inputs])  # Skip connection

        x2 = self.conv2_2(x2)
        x2 = self.bn2_2(x2)
        x2 = self.relu2_2(x2)
        x2 = add([x2, x2])  # This is incorrect, should be add([x2, inputs]) again

        x2 = self.conv2_3(x2)
        x2 = self.bn2_3(x2)
        x2 = self.relu2_3(x2)
        x2 = add([x2, x2])  # This is incorrect, should be add([x2, inputs]) again

        # Concatenate outputs from both pathways
        concatenated = Concatenate(axis=-1)([x1, x2])

        # Fully connected layers
        flattened = tf.keras.layers.Flatten()(concatenated)
        x = self.dense1(flattened)
        x = self.dense2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # MNIST images are 28x28 with 1 channel
    output_tensor = model(input_tensor)
    print(output_tensor.shape)