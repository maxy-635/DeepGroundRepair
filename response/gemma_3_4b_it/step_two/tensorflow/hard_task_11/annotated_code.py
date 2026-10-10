import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(1, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 1),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.concatenate = tf.keras.layers.concatenate([
            self.conv1.output,
            self.conv2.output,
            self.conv3.output
        ])
        self.conv4 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.dense1 = tf.keras.layers.Dense(
            units=10,
            activation='softmax'
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        y = self.conv2(inputs)
        z = self.conv3(inputs)
        merged = self.concatenate([x, y, z])
        merged = self.conv4(merged)
        return self.dense1(merged)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)