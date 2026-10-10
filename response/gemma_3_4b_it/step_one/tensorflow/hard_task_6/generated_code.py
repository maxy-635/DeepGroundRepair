import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', use_bias=False),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', use_bias=False),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', use_bias=False),
            tf.keras.layers.Activation('relu')
        ])
        self.avg_pool = tf.keras.layers.AveragePooling2D((8, 8), strides=(1, 1), padding='valid')
        self.block2 = tf.keras.Sequential([
            tf.keras.layers.Reshape((8, 8, 32, 32), input_shape=(32, 32, 32, 32)),
            tf.keras.layers.Permute([1, 2, 3, 0]),
            tf.keras.layers.Reshape((32, 32, 32), input_shape=(32, 32, 32))
        ])
        self.block3 = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depth_multiplier=1, use_bias=False)
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        main_path_output = self.block1(inputs)
        branch_path_output = self.avg_pool(inputs)
        concatenated = tf.keras.layers.Concatenate(axis=-1)([main_path_output, branch_path_output])
        reshaped = self.block2(concatenated)
        depthwise_conv_output = self.block3(reshaped)
        output = self.dense(depthwise_conv_output)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)