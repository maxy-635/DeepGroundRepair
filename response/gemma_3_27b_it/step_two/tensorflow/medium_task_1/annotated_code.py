import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2d_2 = tf.keras.layers.Conv2D(filters='mask', kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size='mask', strides='mask', padding='mask')
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        x = self.conv2d_1(inputs)  # output tensor shape: (1, 32, 32, 32)
        x = self.conv2d_2(x)  # output tensor shape of this layer should be (1, 32, 32, 3)
        x = self.max_pool(x)  # output tensor shape of this layer should be (1, 32, 32, 64)
        x = tf.keras.layers.Add()([x, inputs])
        x = self.flatten(x)
        x = self.dense_1(x)
        return self.dense_2(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)