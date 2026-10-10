import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=None
        )

        # Branch path
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(
            units=10,
            activation='softmax'
        )

    def call(self, inputs):
        # Main path
        x = self.conv1(inputs)
        x = self.conv2(x)

        # Branch path
        y = self.flatten(inputs)
        y = self.dense(y)

        # Combine paths
        combined = tf.keras.layers.Add()([x, y])

        return combined


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)