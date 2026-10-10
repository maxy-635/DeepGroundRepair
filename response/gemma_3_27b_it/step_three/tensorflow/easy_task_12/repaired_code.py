import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path_block1 = tf.keras.Sequential([
            tf.keras.layers.ReLU(),
            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same'),
        ])
        self.main_path_pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        self.main_path_block2 = tf.keras.Sequential([
            tf.keras.layers.ReLU(),
            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same'),
        ])
        self.main_path_pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        # Branch Path
        self.branch_path_conv = tf.keras.layers.Conv2D(filters=1, kernel_size=(4, 4), strides=(4, 4), padding='valid')

        # Merge and Output
        self.add_layer = tf.keras.layers.Add()
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layer = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        # Main Path
        main_path = self.main_path_block1(inputs)  # output tensor shape: (1, 28, 28, 1)
        main_path = self.main_path_pool1(main_path)  # output tensor shape: (1, 14, 14, 1)
        main_path = self.main_path_block2(main_path)  # output tensor shape: (1, 14, 14, 1)
        main_path = self.main_path_pool2(main_path)  # output tensor shape: (1, 7, 7, 1)

        # Branch Path
        branch_path = self.branch_path_conv(inputs)  # output tensor shape of this layer should be (1, 7, 7, 1)

        # Merge
        merged_path = self.add_layer([main_path, branch_path])

        # Output
        flattened_path = self.flatten_layer(merged_path)
        output = self.dense_layer(flattened_path)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)