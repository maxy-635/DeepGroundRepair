import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Dual-Path Structure
        self.conv1_main = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2_main = tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3_main = tf.keras.layers.Conv2D(1, (3, 3), strides=(1, 1), padding='same', activation='relu')

        self.conv1_branch = tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu')

        self.add = tf.keras.layers.Add()

        # Block 2: Channel Splitting
        self.conv_split1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv_split2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv_split3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')

        self.concat = tf.keras.layers.Concatenate(axis=-1)

        # Fully Connected Layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        # Block 1: Dual-Path
        main_path = self.conv1_main(inputs)
        main_path = self.conv2_main(main_path)
        main_path = self.conv3_main(main_path)
        branch_path = self.conv1_branch(inputs)

        merged = self.add([main_path, branch_path])

        # Block 2: Channel Splitting
        split1 = self.conv_split1(merged)
        split2 = self.conv_split2(merged)
        split3 = self.conv_split3(merged)

        combined = self.concat([split1, split2, split3])

        # Fully Connected Layers
        flattened = self.flatten(combined)
        dense1_output = self.dense1(flattened)
        output = self.dense2(dense1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)