import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_conv1_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.block1_conv1_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.block1_conv1_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')

        # Block 2
        self.block2_reshape = tf.keras.layers.Reshape(target_shape=(32, 32, 3, 32))
        self.block2_permute = tf.keras.layers.Permute(dims='mask')
        self.block2_reshape_back = tf.keras.layers.Reshape(target_shape=(32, 32, 96))

        # Block 3
        self.block3_depthwise = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', activation='relu')

        # Branch Path
        self.branch_pool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')

        # Concatenate
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        # Fully Connected Layer
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main Path
        x1_1 = self.block1_conv1_1(inputs)
        x1_2 = self.block1_conv1_2(inputs)
        x1_3 = self.block1_conv1_3(inputs)
        x1 = tf.keras.layers.Concatenate(axis=-1)([x1_1, x1_2, x1_3])

        x2 = self.block2_reshape(x1)
        x2 = self.block2_permute(x2)
        x2 = self.block2_reshape_back(x2)

        x3 = self.block3_depthwise(x2)

        # Branch Path
        y = self.branch_pool(inputs)
        y = tf.keras.layers.Reshape((32, 32, 96))(y)

        # Concatenation
        combined = self.concatenate([x3, y])

        # Fully Connected Layer
        output = self.dense(combined)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)