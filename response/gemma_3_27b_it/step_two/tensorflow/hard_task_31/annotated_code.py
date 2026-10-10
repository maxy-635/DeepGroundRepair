import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Main and Branch Paths
        self.main_path = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.Dropout(rate=0.2)
        ])
        self.restore_channels = tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        
        # Block 2: Separable Convolution Groups
        self.group1 = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.Dropout(rate=0.2)
        ])
        self.group2 = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.Dropout(rate=0.2)
        ])
        self.group3 = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.Dropout(rate=0.2)
        ])

        # Final Layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1
        main_path_output = self.main_path(inputs)
        main_path_output = self.restore_channels(main_path_output)
        branch_path_output = inputs
        block1_output = tf.keras.layers.Add()([main_path_output, branch_path_output])

        # Block 2
        split_input = tf.split(block1_output, num_or_size_splits=3, axis=-1)
        group1_output = self.group1(split_input[0])
        group2_output = self.group2(split_input[1])
        group3_output = self.group3(split_input[2])
        block2_output = tf.keras.layers.Concatenate(axis=-1)([group1_output, group2_output, group3_output])

        # Final Layers
        flattened_output = self.flatten(block2_output)
        output = self.dense(flattened_output)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)