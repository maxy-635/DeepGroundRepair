import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn1 = BatchNormalization(axis=-1)
        self.relu1 = ReLU()

        # Block 2
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn2 = BatchNormalization(axis=-1)
        self.relu2 = ReLU()

        # Block 3
        self.conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn3 = BatchNormalization(axis=-1)
        self.relu3 = ReLU()

        # Concatenation
        self.concat = Concatenate(axis=-1)

        # Fully connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(units=256, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        block1_output = x

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        block2_output = x

        # Concatenate block1_output and block2_output
        x = self.concat([block1_output, block2_output])

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        block3_output = x

        # Concatenate block2_output and block3_output
        x = self.concat([block2_output, block3_output])

        # Flatten and fully connected layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Should print (None, 10)