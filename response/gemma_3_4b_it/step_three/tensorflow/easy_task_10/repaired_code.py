import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(1, 1),
            strides=(2, 2),
            padding='same'
        )
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            depth_multiplier=1
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(1, 1),
            strides=(2, 2),
            padding='same'
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(
            units=10,
            activation='softmax'
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.depthwise_conv(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)