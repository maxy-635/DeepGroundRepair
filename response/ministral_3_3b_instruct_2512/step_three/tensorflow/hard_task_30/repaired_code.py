import tensorflow as tf

class DL_Model:
    def __init__(self):
        self.model = tf.keras.Sequential()

    def __call__(self, input_tensor):
        # First block: Dual-path structure
        main_path = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=None)
        ])

        # Apply branch path to the output of the first convolution in main_path
        branch_path = tf.keras.layers.Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None)(main_path[0](input_tensor))

        # Combine paths
        combined = tf.keras.layers.Add()([main_path[1](input_tensor), branch_path])

        # Second block: Depthwise separable convolutions
        # Split the combined tensor into three groups along the channel dimension
        split = tf.keras.layers.Lambda(lambda x: tf.split(x, 3, axis=3))(combined)

        # Process each group with depthwise separable convolutions
        conv_1x1 = tf.keras.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        ])
        conv_3x3 = tf.keras.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        ])
        conv_5x5 = tf.keras.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='valid', depth_multiplier=1, activation='relu')
        ])

        processed = tf.keras.layers.Concatenate(axis=3)([
            conv_1x1(split[0]),
            conv_3x3(split[1]),
            conv_5x5(split[2])
        ])

        # Flatten and dense layers
        flattened = tf.keras.layers.Flatten()(processed)
        dense1 = tf.keras.layers.Dense(128, activation='relu')(flattened)
        dense2 = tf.keras.layers.Dense(10, activation='softmax')(dense1)

        return dense2

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)