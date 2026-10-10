import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.initial_conv = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
        )
        self.split_layer = tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))
        self.conv1x1_1 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu',
        )
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            depth_multiplier=1,
            activation='relu',
        )
        self.conv1x1_2 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu',
        )
        self.concat_layer = tf.keras.layers.Concatenate(axis=-1)

        # Block 2
        self.shape_layer = tf.keras.layers.Lambda(lambda x: tf.shape(x))
        self.reshape_layer = tf.keras.layers.Reshape((-1, -1, 4, 8))
        self.permute_layer = tf.keras.layers.Permute((0, 1, 3, 2))
        self.reshape_layer2 = tf.keras.layers.Reshape((-1, -1, 32))

        self.flatten = tf.keras.layers.Flatten()
        self.dense_layer = tf.keras.layers.Dense(
            units=10,
            activation='softmax',
        )

    def call(self, inputs):
        # Block 1
        x = self.initial_conv(inputs)
        split_x = self.split_layer(x)
        group1 = split_x[0]
        group2 = split_x[1]

        x1 = self.conv1x1_1(group1)
        x2 = self.depthwise_conv(x1)
        x3 = self.conv1x1_2(x2)

        merged = self.concat_layer([x3, group2])

        # Block 2
        shape = self.shape_layer(merged)
        height = shape[1]
        width = shape[2]
        channels = shape[3]
        groups = 4
        channels_per_group = channels // groups
        reshaped = self.reshape_layer(merged)
        permuted = self.permute_layer(reshaped)
        reshaped2 = self.reshape_layer2(permuted)

        flattened = self.flatten(reshaped2)
        output = self.dense_layer(flattened)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)