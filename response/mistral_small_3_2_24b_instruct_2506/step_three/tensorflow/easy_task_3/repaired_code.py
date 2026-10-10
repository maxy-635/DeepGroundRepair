import tensorflow as tf
from tensorflow.keras import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: two convolutions followed by max pooling
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')
        self.max_pool1 = tf.raw_ops.MaxPoolV2(input=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Second block: two convolutions followed by max pooling
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')
        self.conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')
        self.max_pool2 = tf.raw_ops.MaxPoolV2(input=None, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Third block: two convolutions followed by max pooling
        self.conv5 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')
        self.conv6 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')
        self.max_pool3 = tf.raw_ops.MaxPoolV2(input=None, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Flatten and fully connected layers
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(units=128, activation='relu')
        self.fc3 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.max_pool1(x)

        # Second block
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool2(x)

        # Third block
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.max_pool3(x)

        # Flatten and fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # MNIST image shape: 28x28 with 1 channel
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (1, 10)