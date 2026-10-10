import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial Convolutional Layer
        self.conv_initial = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'
        )

        # Block 1
        self.conv1x1_1 = tf.keras.layers.Conv2D(
            filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'
        )
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu'
        )
        self.conv1x1_2 = tf.keras.layers.Conv2D(
            filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'
        )
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        # Block 2
        self.reshape = tf.keras.layers.Reshape(target_shape=(None, None, 4, 8))
        self.permute = tf.keras.layers.Permute(dims='mask')
        self.reshape_back = tf.keras.layers.Reshape(target_shape=(None, None, 32))

        # Classification Layer
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Initial Convolution
        x = self.conv_initial(inputs)

        # Block 1
        group1 = self.conv1x1_1(x)
        group1 = self.depthwise_conv(group1)
        group1 = self.conv1x1_2(group1)
        group2 = x
        x = self.concatenate([group1, group2])

        # Block 2
        shape = tf.shape(x)
        height = shape[1]
        width = shape[2]
        channels = shape[3]

        x = self.reshape(x)
        x = self.permute(x)
        x = self.reshape_back(x)

        # Classification
        x = self.flatten(x)
        x = self.dense(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)