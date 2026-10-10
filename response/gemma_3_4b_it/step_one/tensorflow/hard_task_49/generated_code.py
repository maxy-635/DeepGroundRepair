import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Average pooling
        self.avg_pool1 = tf.keras.layers.AveragePooling1D(pool_size=1, strides=1, padding='VALID')
        self.avg_pool2 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='VALID')
        self.avg_pool3 = tf.keras.layers.AveragePooling1D(pool_size=4, strides=4, padding='VALID')

        # Flatten and concatenate
        self.flatten = tf.keras.layers.Flatten()
        self.concat1 = tf.keras.layers.Concatenate(axis=-1)

        # Intermediate dense layer and reshape
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.reshape = tf.keras.layers.Reshape((4, 128))

        # Second block: Depthwise separable convolutions
        self.conv1 = tf.keras.layers.DepthwiseConv2D(kernel_size=1, strides=1, padding='VALID', depth_multiplier=1, activation='relu')
        self.conv2 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='VALID', depth_multiplier=1, activation='relu')
        self.conv3 = tf.keras.layers.DepthwiseConv2D(kernel_size=5, strides=1, padding='VALID', depth_multiplier=1, activation='relu')
        self.conv4 = tf.keras.layers.DepthwiseConv2D(kernel_size=7, strides=1, padding='VALID', depth_multiplier=1, activation='relu')

        # Concatenate the outputs
        self.concat2 = tf.keras.layers.Concatenate(axis=-1)

        # Final dense layer
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block
        x = self.avg_pool1(inputs)
        x = self.avg_pool2(x)
        x = self.avg_pool3(x)
        x = self.flatten(x)

        # Intermediate layer
        x = self.dense1(x)
        x = self.reshape(x)

        # Second block
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.concat2(x)

        # Final layer
        x = self.dense2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Example input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)