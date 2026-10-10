import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path_block1 = self.create_main_path_block()
        self.main_path_block2 = self.create_main_path_block()
        self.main_path_block3 = self.create_main_path_block()

        # Branch path
        self.branch_path_conv = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Fusion and classification
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def create_main_path_block(self):
        return tf.keras.Sequential([
            tf.keras.layers.ReLU(),
            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1),
            tf.keras.layers.Concatenate(axis=-1)
        ])

    def call(self, inputs):
        # Main path
        main_path_output = inputs
        for block in [self.main_path_block1, self.main_path_block2, self.main_path_block3]:
            main_path_output = block(main_path_output)

        # Branch path
        branch_path_output = self.branch_path_conv(inputs)

        # Fusion
        fused_output = tf.keras.layers.Add()([main_path_output, branch_path_output])

        # Classification
        flattened_output = self.flatten(fused_output)
        output = self.dense(flattened_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)