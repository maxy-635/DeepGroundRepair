import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            depth_multiplier=1,
            activation='relu'
        )
        self.dropout1 = tf.keras.layers.Dropout(rate=0.2)
        self.conv1x1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.dropout2 = tf.keras.layers.Dropout(rate=0.2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(
            units=10,
            activation='softmax'
        )

    def call(self, inputs):
        x = self.depthwise_conv(inputs)
        x = self.dropout1(x)
        x = self.conv1x1(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        return self.dense(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)