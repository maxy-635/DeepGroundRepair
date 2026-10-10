import tensorflow as tf
from tensorflow.keras.layers import Layer

class DL_Model(Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First convolutional layer
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')
        # Second convolutional layer
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')
        # Max-pooling layer
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        # Flatten layer
        self.flatten = tf.keras.layers.Flatten()
        # First fully connected layer
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        # Second fully connected layer (output layer)
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First convolutional layer
        x = self.conv1(inputs)
        # Second convolutional layer
        x = self.conv2(x)
        # Max-pooling layer
        x = self.max_pool(x)
        # Remove the problematic addition operation
        # Flatten the features
        x = self.flatten(x)
        # First fully connected layer
        x = self.dense1(x)
        # Second fully connected layer (output layer)
        x = self.dense2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print(output_tensor)