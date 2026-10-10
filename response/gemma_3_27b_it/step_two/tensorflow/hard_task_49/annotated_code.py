import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.avg_pool_1 = tf.keras.layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.avg_pool_2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.avg_pool_3 = tf.keras.layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
        self.flatten_1 = tf.keras.layers.Flatten()
        self.flatten_2 = tf.keras.layers.Flatten()
        self.flatten_3 = tf.keras.layers.Flatten()
        self.concat = tf.keras.layers.Concatenate(axis=1)

        # Transition layer
        self.dense = tf.keras.layers.Dense(units=64, activation='relu')
        self.reshape = tf.keras.layers.Reshape(target_shape='mask')

        # Block 2
        self.split = tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))

        self.depthwise_1 = tf.keras.layers.DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.separable_1 = tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.separable_2 = tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.separable_3 = tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')

        self.concat_2 = tf.keras.layers.Concatenate(axis=-1)
        self.flatten_4 = tf.keras.layers.Flatten()
        self.dense_2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        # Block 1
        x1 = self.avg_pool_1(inputs)  # output tensor shape: (1, 28, 28, 1)
        x2 = self.avg_pool_2(inputs)  # output tensor shape: (1, 14, 14, 1)
        x3 = self.avg_pool_3(inputs)  # output tensor shape: (1, 7, 7, 1)

        x1_flat = self.flatten_1(x1)  # output tensor shape: (1, 784)
        x2_flat = self.flatten_2(x2)  # output tensor shape: (1, 196)
        x3_flat = self.flatten_3(x3)  # output tensor shape: (1, 49)

        x_concat = self.concat([x1_flat, x2_flat, x3_flat])  # output tensor shape: (1, 1029)

        # Transition layer
        x = self.dense(x_concat)  # output tensor shape: (1, 64)
        x = self.reshape(x)

        # Block 2
        splits = self.split(x)
        x1_block2 = self.depthwise_1(splits[0])
        x2_block2 = self.separable_1(splits[1])
        x3_block2 = self.separable_2(splits[2])
        x4_block2 = self.separable_3(splits[3])

        x_concat_2 = self.concat_2([x1_block2, x2_block2, x3_block2, x4_block2])
        x_flat = self.flatten_4(x_concat_2)
        output = self.dense_2(x_flat)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)