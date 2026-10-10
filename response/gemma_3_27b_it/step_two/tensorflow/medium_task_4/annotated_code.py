import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Path 1
        self.conv1_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv1_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.avg_pool1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        # Path 2
        self.conv2_1 = tf.keras.layers.Conv2D(filters='mask', kernel_size='mask', strides='mask', padding='mask', activation='relu')

        # Flatten and Dense layer
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Path 1
        path1 = self.conv1_1(inputs)  # output tensor shape: (1, 32, 32, 32)
        path1 = self.conv1_2(path1)  # output tensor shape: (1, 32, 32, 64)
        path1 = self.avg_pool1(path1)  # output tensor shape: (1, 16, 16, 64)

        # Path 2
        path2 = self.conv2_1(inputs)  # output tensor shape of this layer should be (1, 16, 16, 64)

        # Combine paths
        combined = tf.add(path1, path2)

        # Flatten and Dense layer
        flattened = self.flatten(combined)
        output = self.dense(flattened)
        return output


if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 color channels
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)