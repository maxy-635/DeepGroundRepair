import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims='mask')
        self.dense1 = tf.keras.layers.Dense(units=32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units='mask', activation='relu')
        self.reshape = tf.keras.layers.Reshape((1, 1, 32))
        self.multiply = tf.keras.layers.Multiply()
        self.flatten = tf.keras.layers.Flatten()
        self.dense3 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        x = self.global_avg_pool(inputs)  # output tensor shape of this layer should be (1, 32, 32, 3)
        x = self.dense1(x)
        x = self.dense2(x)  # output tensor shape of this layer should be (1, 1, 1, 3)
        x = self.reshape(x)
        x = self.multiply([inputs, x])
        x = self.flatten(x)
        return self.dense3(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)