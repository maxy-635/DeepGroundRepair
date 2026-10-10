import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.nn.leaky_relu
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='same',
            activation=tf.nn.leaky_relu
        )
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.dense1 = tf.keras.layers.Dense(
            units=128,
            activation=tf.nn.leaky_relu
        )
        self.dense2 = tf.keras.layers.Dense(
            units=10,
            activation=tf.keras.layers.Softmax()
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = tf.keras.layers.Dense(units=64)(x)
        x = tf.keras.layers.Dense(units=64)(x)
        x = tf.keras.layers.Dense(units=10, activation=tf.keras.layers.Softmax())(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)