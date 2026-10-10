import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')

        # Branch path
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size='mask', strides='mask', padding='mask', activation='relu')

        # Concatenation and fully connected layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Main path
        x1 = self.conv1(inputs)  # output tensor shape: (1, 32, 32, 32)
        x1 = self.conv2(x1)  # output tensor shape: (1, 32, 32, 64)
        x1 = self.maxpool(x1)  # output tensor shape: (1, 16, 16, 64)

        # Branch path
        x2 = self.conv3(inputs)  # output tensor shape of this layer should be (1, 16, 16, 32)

        # Concatenate the outputs of both paths
        x = tf.keras.layers.Concatenate(axis=-1)([x1, x2])

        # Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Batch size of 1, 32x32 RGB images
    output_tensor = model(input_tensor)
    print(output_tensor.shape)