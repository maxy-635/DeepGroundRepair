import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')
        self.conv2_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.conv2_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (32, 32, 64)
        x = self.conv1(inputs)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = tf.keras.layers.concatenate([x, x], axis=-1)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((32, 32, 64))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)