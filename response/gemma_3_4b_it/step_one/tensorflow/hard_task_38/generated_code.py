import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv_block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same',
                                  kernel_initializer='he_normal', use_bias=False),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU(max_value=0.2)
        ])
        self.conv_block2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same',
                                  kernel_initializer='he_normal', use_bias=False),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU(max_value=0.2)
        ])
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        path1 = self.conv_block1(inputs)
        path2 = self.conv_block2(inputs)

        merged = tf.keras.layers.Concatenate(axis=-1)([path1, path2])
        merged = self.flatten(merged)
        merged = self.dense1(merged)
        outputs = self.dense2(merged)

        return outputs


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)