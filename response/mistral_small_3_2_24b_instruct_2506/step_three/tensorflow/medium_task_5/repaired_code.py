import tensorflow as tf
from tensorflow.keras.layers import Layer

class DL_Model(Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.conv1_main = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')
        self.conv2_main = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')
        self.maxpool_main = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Branch path
        self.conv1_branch = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')
        self.maxpool_branch = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Fully connected layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main path
        x_main = self.conv1_main(inputs)
        x_main = self.conv2_main(x_main)
        x_main = self.maxpool_main(x_main)

        # Branch path
        x_branch = self.conv1_branch(inputs)
        x_branch = self.maxpool_branch(x_branch)

        # Combine paths
        x_combined = tf.math.add(x_main, x_branch)

        # Fully connected layers
        x_flat = self.flatten(x_combined)
        x_dense = self.dense1(x_flat)
        output = self.dense2(x_dense)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)