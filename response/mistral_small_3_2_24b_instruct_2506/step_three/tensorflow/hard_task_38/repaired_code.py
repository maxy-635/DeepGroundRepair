import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Concatenate

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First pathway
        self.conv1a = Conv2D(filters=32, kernel_size=3, padding='same')
        self.bn1a = BatchNormalization(axis=-1)
        self.conv1b = Conv2D(filters=32, kernel_size=3, padding='same')
        self.bn1b = BatchNormalization(axis=-1)
        self.conv1c = Conv2D(filters=32, kernel_size=3, padding='same')
        self.bn1c = BatchNormalization(axis=-1)

        # Second pathway
        self.conv2a = Conv2D(filters=32, kernel_size=3, padding='same')
        self.bn2a = BatchNormalization(axis=-1)
        self.conv2b = Conv2D(filters=32, kernel_size=3, padding='same')
        self.bn2b = BatchNormalization(axis=-1)
        self.conv2c = Conv2D(filters=32, kernel_size=3, padding='same')
        self.bn2c = BatchNormalization(axis=-1)

        # Concatenation and classification layers
        self.concat = Concatenate(axis=-1)
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        # First pathway
        x1 = inputs  # output tensor shape: (1, 28, 28, 1)
        for _ in range(3):  # output tensor shape: (1)
            x1 = self.bn1a(x1)  # output tensor shape: (1, 28, 28, 1)
            x1 = tf.nn.relu(x1)  # output tensor shape: (1, 28, 28, 1)
            x1 = self.conv1a(x1)  # output tensor shape: (1, 28, 28, 32)
            x1 = tf.experimental.numpy.concatenate([inputs, x1], axis=-1)  # output tensor shape: (1, 28, 28, 33)

        # Second pathway
        x2 = inputs
        for _ in range(3):
            x2 = self.bn2a(x2)
            x2 = tf.nn.relu(x2)
            x2 = self.conv2a(x2)
            x2 = tf.experimental.numpy.concatenate([inputs, x2], axis=-1)

        # Concatenate both pathways
        x = self.concat([x1, x2])

        # Classification
        x = tf.keras.layers.Flatten()(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal([1, 28, 28, 1])  # MNIST image shape
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)