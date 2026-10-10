import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1
        self.avg_pool1 = tf.keras.layers.AveragePooling1D(pool_size='mask', strides='mask')
        self.avg_pool2 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)
        self.avg_pool3 = tf.keras.layers.AveragePooling1D(pool_size=4, strides=4)
        self.dropout1 = tf.keras.layers.Dropout(rate=0.5)
        self.flatten1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.reshape1 = tf.keras.layers.Reshape(target_shape=(4, 128))

        # Block 2
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')
        self.avg_pool4 = tf.keras.layers.AveragePooling1D(pool_size=8, strides=8)
        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        # Block 1
        x = self.avg_pool1(inputs)
        x = self.dropout1(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.reshape1(x)

        # Block 2
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        z = self.avg_pool4(y)
        z = self.concat([z, x])
        output = self.dense2(z)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Example input shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)