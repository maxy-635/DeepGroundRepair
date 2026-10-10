import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.conv1x1_main = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.conv1x1_parallel = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.conv3x3_parallel = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        # Branch Path
        self.conv3x3_branch = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')

        # Combine Paths
        self.add = tf.keras.layers.Add()

        # Classification
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')  # Assuming 10 classes

    def call(self, inputs):
        # Main Path
        x_main = self.conv1x1_main(inputs)
        x_parallel1 = self.conv1x1_parallel(x_main)
        x_parallel2 = self.conv3x3_parallel(x_main)
        x_main = self.concatenate([x_parallel1, x_parallel2])

        # Branch Path
        x_branch = self.conv3x3_branch(inputs)

        # Combine Paths
        x = self.add([x_main, x_branch])

        # Classification
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 64))  # Batch size of 1
    output_tensor = model(input_tensor)
    print(output_tensor.shape)