import tensorflow as tf
from tensorflow.keras import layers

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.conv1 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
        self.conv2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.conv3 = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')
        self.max_pool = layers.MaxPooling2D(pool_size=(2, 2))

        # Branch path
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(units=64, activation='relu')
        self.dense2 = layers.Dense(units=128, activation='relu')

        # Final layers
        self.fc1 = layers.Dense(units=128, activation='relu')
        self.fc2 = layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main path
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x_main = self.max_pool(x)

        # Branch path
        x_branch = self.global_avg_pool(inputs)
        x_branch = self.dense1(x_branch)
        x_branch = self.dense2(x_branch)
        x_branch = tf.reshape(x_branch, (-1, 1, 1, 128))
        x_branch = tf.math.multiply(x_main, x_branch)

        # Combine paths
        x_combined = tf.math.add(x_main, x_branch)
        x_combined = tf.keras.layers.Flatten()(x_combined)
        x_combined = self.fc1(x_combined)
        output = self.fc2(x_combined)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)