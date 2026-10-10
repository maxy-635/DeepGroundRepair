import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=7,
            strides=1,
            padding='same',
            depth_multiplier=1,
            activation=None
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6)
        self.pointwise_conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=None
        )
        self.pointwise_conv2 = tf.keras.layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same', activation=None)







        # Branch Path
        # No layers in the branch path, it's just a direct connection

        # Combination and Classification
        self.add_layer = tf.keras.layers.Add()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Main Path
        main_path = self.depthwise_conv(inputs)  # output tensor shape: (1, 32, 32, 3)
        main_path = self.layer_norm(main_path)  # output tensor shape: (1, 32, 32, 3)
        main_path = self.pointwise_conv1(main_path)  # output tensor shape: (1, 32, 32, 32)
        main_path = self.pointwise_conv2(main_path)  # output tensor shape of this layer should be (1, 32, 32, 3)

        # Branch Path
        branch_path = inputs  # output tensor shape: (1, 32, 32, 3)

        # Combine Paths
        combined_output = self.add_layer([main_path, branch_path])

        # Classification
        flattened_output = self.flatten(combined_output)
        x = self.dense1(flattened_output)
        output = self.dense2(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # CIFAR-10 input shape: (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)