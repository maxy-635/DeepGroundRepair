import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Dual-Path Structure
        self.main_path = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        ])
        self.branch_path = tf.keras.layers.Identity()

        # Block 2: Depthwise Separable Convolutions
        self.depthwise_group1 = tf.keras.layers.DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.depthwise_group2 = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.depthwise_group3 = tf.keras.layers.DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')

        # Fully Connected Layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1
        main_path_output = self.main_path(inputs)
        branch_path_output = self.branch_path(inputs)
        block1_output = tf.keras.layers.Add()([main_path_output, branch_path_output])

        # Block 2
        channel_split = tf.split(block1_output, num_or_size_splits=3, axis=-1)
        group1_output = self.depthwise_group1(channel_split[0])
        group2_output = self.depthwise_group2(channel_split[1])
        group3_output = self.depthwise_group3(channel_split[2])
        block2_output = tf.keras.layers.Concatenate(axis=-1)([group1_output, group2_output, group3_output])

        # Fully Connected Layers
        x = self.flatten(block2_output)
        x = self.dense1(x)
        output = self.dense2(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)