import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.reshape_layer = tf.keras.layers.Reshape((128,))
        self.flatten = tf.keras.layers.Flatten()
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        x = self.global_avg_pool(inputs)  # output tensor shape: (1, 3)
        x = self.dense1(x)  # output tensor shape: (1, 128)
        x = self.reshape_layer(x)  # output tensor shape: (1, 128)
        x = tf.multiply(x, inputs)
        x = self.flatten(x)
        x = self.dense2(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)