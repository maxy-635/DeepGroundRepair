import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=1, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=1, padding='same')
        self.concatenation = tf.keras.layers.concatenate([self.conv1.output, self.conv2.output, self.conv3.output], axis=-1)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.concatenation
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)