import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu1 = tf.keras.layers.ReLU()

        # Global average pooling and fully connected layers
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(units=32, activation=None)
        self.dense2 = tf.keras.layers.Dense(units='mask', activation=None)

        # Reshape and multiply layers
        self.reshape = tf.keras.layers.Reshape(target_shape=(1, 1, 32))
        self.multiply = tf.keras.layers.Multiply()

        # Concatenate layer
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        # Downsampling layers
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None)
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        # Final fully connected layer
        self.flatten = tf.keras.layers.Flatten()
        self.dense3 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Initial feature extraction
        x = self.conv1(inputs)  # output tensor shape: (1, 32, 32, 32)
        x = self.bn1(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.relu1(x)  # output tensor shape: (1, 32, 32, 32)

        # Global average pooling and fully connected layers
        x = self.gap(x)  # output tensor shape of this layer should be (1, 32)
        x = self.dense1(x)
        x = self.dense2(x)  # output tensor shape of this layer should be (1, 32)

        # Reshape and multiply
        x = self.reshape(x)
        x = self.multiply([x, inputs])

        # Concatenate
        x = self.concatenate([inputs, x])

        # Downsampling
        x = self.conv2(x)
        x = self.avgpool(x)

        # Flatten and final classification
        x = self.flatten(x)
        x = self.dense3(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)