import tensorflow as tf
from tensorflow.keras.layers import Layer

class DL_Model(Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path layers
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')
        # Branch path (identity connection)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main path
        x = self.conv1(inputs)
        x = self.conv2(x)
        # Branch path
        branch = inputs
        # Combine paths
        combined = tf.math.add(x, branch)
        # Final layers
        x = self.flatten(combined)
        x = self.dense(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # MNIST image shape
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)