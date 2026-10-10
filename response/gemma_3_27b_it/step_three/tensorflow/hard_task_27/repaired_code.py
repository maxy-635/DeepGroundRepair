import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=7,
            strides=1,
            padding='same',
            depth_multiplier=1,
            activation=tf.keras.activations.relu
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.dense1 = tf.keras.layers.Dense(
            units=32,  # Assuming 3 input channels, adjust if different
            activation=tf.keras.activations.relu
        )
        self.dense2 = tf.keras.layers.Dense(
            units=32,  # Assuming 3 input channels, adjust if different
            activation=tf.keras.activations.relu
        )
        self.dense3 = tf.keras.layers.Dense(
            units=10,
            activation='softmax'
        )

    def call(self, inputs):
        x = self.depthwise_conv(inputs)
        x = self.layer_norm(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = tf.keras.layers.Add()([inputs, x])
        return self.dense3(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)