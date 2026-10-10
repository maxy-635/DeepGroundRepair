import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same',
                                   kernel_initializer='he_normal', use_bias=False),
            tf.keras.layers.Conv2D(filters='mask', kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=False),

            tf.keras.layers.AveragePooling2D(pool_size='mask', strides='mask', padding='mask')
        ])
        self.block2 = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(keepdims=True),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        x = self.block1(inputs)  # output tensor shape of this layer should be (1, 16, 16, 32)
        x = inputs + x
        x = self.block2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)