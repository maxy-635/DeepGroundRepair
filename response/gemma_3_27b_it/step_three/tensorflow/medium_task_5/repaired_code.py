import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.main_conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.main_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Branch Path
        self.branch_conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Combine Paths
        self.add_layer = tf.keras.layers.Add()

        # Classification
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Main Path
        main_path = self.main_conv1(inputs)  # output tensor shape: (1, 32, 32, 32)
        main_path = self.main_conv2(main_path)  # output tensor shape: (1, 32, 32, 64)
        main_path = self.main_pool(main_path)  # output tensor shape: (1, 16, 16, 64)

        # Branch Path
        branch_path = self.branch_conv1(inputs)  # output tensor shape of this layer should be (1, 32, 32, 64)
        branch_path = self.branch_pool(branch_path)

        # Combine Paths
        combined_path = self.add_layer([main_path, branch_path])

        # Classification
        flattened = self.flatten(combined_path)
        dense1_output = self.dense1(flattened)
        output = self.dense2(dense1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels
    output_tensor = model(input_tensor)
    print(output_tensor.shape)