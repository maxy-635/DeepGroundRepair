import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.conv1_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.conv1_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.concat1 = tf.keras.layers.Concatenate(axis=-1)

        # Block 2
        self.reshape1 = tf.keras.layers.Reshape(target_shape=(-1, 3, 32))
        self.permute1 = tf.keras.layers.Permute(dims=[0, 2, 1, 3])
        self.reshape2 = tf.keras.layers.Reshape(target_shape=(-1, 32, 32, 3))

        # Block 3
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', activation='relu')

        # Branch
        self.identity = tf.keras.layers.Identity()

        # Addition
        self.add1 = tf.keras.layers.Add()

        # Fully Connected Layer
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1
        x1_1 = self.conv1_1(inputs)
        x1_2 = self.conv1_2(inputs)
        x1_3 = self.conv1_3(inputs)
        x1 = self.concat1([x1_1, x1_2, x1_3])

        # Block 2
        x2_reshape = self.reshape1(x1)
        x2_permute = self.permute1(x2_reshape)
        x2 = self.reshape2(x2_permute)

        # Block 3
        x3 = self.depthwise_conv(x2)

        # Branch
        identity = self.identity(inputs)

        # Addition
        x_add = self.add1([x3, identity])

        # Fully Connected Layer
        x_flatten = self.flatten(x_add)
        output = self.dense1(x_flatten)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)