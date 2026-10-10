import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.conv1x1 = tf.keras.layers.DepthwiseConv2D(kernel_size=1, strides=1, padding='same', activation='relu')
        self.conv3x3 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv5x5 = tf.keras.layers.DepthwiseConv2D(kernel_size=5, strides=1, padding='same', activation='relu')

        # Branch Path
        self.branch_conv = tf.keras.layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same', activation='relu')

        # Concatenate and Add
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)
        self.add = tf.keras.layers.Add()

        # Classification
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Split input channels
        num_channels = inputs.shape[-1]  # output tensor shape: (1)
        group1 = inputs[:, :, :, :num_channels//3]  # output tensor shape: (1, 32, 32, 1)
        group2 = inputs[:, :, :, num_channels//3:2*num_channels//3]  # output tensor shape: (1, 32, 32, 1)
        group3 = inputs[:, :, :, 2*num_channels//3:]  # output tensor shape: (1, 32, 32, 1)

        # Main Path
        main_path_output1 = self.conv1x1(group1)  # output tensor shape: (1, 32, 32, 1)
        main_path_output2 = self.conv3x3(group2)  # output tensor shape: (1, 32, 32, 1)
        main_path_output3 = self.conv5x5(group3)  # output tensor shape: (1, 32, 32, 1)

        main_path_output = self.concatenate([main_path_output1, main_path_output2, main_path_output3])  # output tensor shape: (1, 32, 32, 3)

        # Branch Path
        branch_path_output = self.branch_conv(inputs)  # output tensor shape of this layer should be (1, 32, 32, 3)

        # Add Main and Branch Paths
        added_output = self.add([main_path_output, branch_path_output])

        # Classification
        flattened_output = self.flatten(added_output)
        dense1_output = self.dense1(flattened_output)
        output = self.dense2(dense1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)