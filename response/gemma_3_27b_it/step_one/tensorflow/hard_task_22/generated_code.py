import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.conv1_1 = tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')
        self.conv1_3 = tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')
        self.conv1_5 = tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')

        # Branch Path
        self.branch_conv = tf.keras.layers.Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')

        # Fusion
        self.add_layer = tf.keras.layers.Add()

        # Classification
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main Path
        group1 = self.conv1_1(inputs)
        group2 = self.conv1_3(inputs)
        group3 = self.conv1_5(inputs)
        main_path_output = tf.keras.layers.Concatenate(axis=-1)([group1, group2, group3])

        # Branch Path
        branch_path_output = self.branch_conv(inputs)

        # Fusion
        fused_output = self.add_layer([main_path_output, branch_path_output])

        # Classification
        flattened_output = self.flatten_layer(fused_output)
        dense1_output = self.dense1(flattened_output)
        output = self.dense2(dense1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)