import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial channel expansion
        self.channel_expand = tf.keras.layers.Conv2D(
            filters=32, kernel_size=1, strides=1, padding="same", activation=None
        )

        # Depthwise separable convolution
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3, strides=1, padding="same", activation=None
        )

        # Channel attention mechanism
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.dense1 = tf.keras.layers.Dense(units=16, activation="relu")
        self.dense2 = tf.keras.layers.Dense(units=32, activation="sigmoid")
        self.reshape = tf.keras.layers.Reshape(target_shape=(1, 1, 32))
        self.multiply = tf.keras.layers.Multiply()

        # Dimensionality reduction and skip connection
        self.channel_reduce = tf.keras.layers.Conv2D(filters='mask', kernel_size=1, strides=1, padding='same', activation=None)


        self.add = tf.keras.layers.Add()

        # Classification layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense_classify = tf.keras.layers.Dense(units=10, activation="softmax")

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Initial channel expansion
        x = self.channel_expand(inputs)  # output tensor shape: (1, 32, 32, 32)

        # Depthwise separable convolution
        x = self.depthwise_conv(x)  # output tensor shape: (1, 32, 32, 32)

        # Channel attention
        attention = self.global_avg_pool(x)  # output tensor shape: (1, 1, 1, 32)
        attention = self.dense1(attention)  # output tensor shape: (1, 1, 1, 16)
        attention = self.dense2(attention)  # output tensor shape: (1, 1, 1, 32)
        attention = self.reshape(attention)  # output tensor shape: (1, 1, 1, 32)
        x = self.multiply([x, attention])  # output tensor shape: (1, 32, 32, 32)

        # Dimensionality reduction and skip connection
        x = self.channel_reduce(x)  # output tensor shape of this layer should be (1, 32, 32, 3)
        x = self.add([x, inputs])

        # Classification
        x = self.flatten(x)
        x = self.dense_classify(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)