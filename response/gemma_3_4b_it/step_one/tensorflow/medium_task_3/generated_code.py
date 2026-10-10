import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.pool1 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.pool2 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid'
        )
        self.concat = tf.keras.layers.concatenate([
            self.pool1.output,
            self.pool2.output
        ], axis=-1)
        self.dense = tf.keras.layers.Dense(
            units=10,
            activation='softmax'
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.concat([inputs, x])
        x = self.dense(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)