import tensorflow as tf
from tensorflow.keras.layers import Layer

class DL_Model(Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define three convolutional layers
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')

        # Define fully connected layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation=None)  # No activation for logits

    def call(self, inputs):
        # First convolutional layer
        x = self.conv1(inputs)

        # Second convolutional layer with concatenation
        x = tf.experimental.numpy.concatenate([inputs, x], axis=-1)
        x = self.conv2(x)

        # Third convolutional layer with concatenation
        x = tf.experimental.numpy.concatenate([x, inputs], axis=-1)
        x = self.conv3(x)

        # Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        # Apply softmax for classification probabilities
        output = tf.nn.softmax(x, axis=-1)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print(output_tensor)