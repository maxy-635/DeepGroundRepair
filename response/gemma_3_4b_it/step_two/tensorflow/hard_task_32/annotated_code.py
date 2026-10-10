import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.keras.activations.relu
        )
        self.depthwise_conv1 = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            depth_multiplier=1,
            activation=tf.keras.activations.relu
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.keras.activations.relu
        )
        self.depthwise_conv2 = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            depth_multiplier=1,
            activation=tf.keras.activations.relu
        )
        self.dropout1 = tf.keras.layers.Dropout(rate=0.25)
        self.dropout2 = tf.keras.layers.Dropout(rate=0.25)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            units=128,
            activation=tf.keras.activations.relu
        )
        self.dense2 = tf.keras.layers.Dense(
            units=10,
            activation=tf.keras.activations.softmax
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.dropout1(x)
        x = self.depthwise_conv1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.depthwise_conv2(x)

        x = tf.concat([x, x], axis=-1)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)