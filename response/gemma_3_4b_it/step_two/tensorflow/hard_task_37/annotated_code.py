import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1_block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', 
                                   activation='relu'),
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', 
                                   activation='relu'),
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', 
                                   activation='relu')
        ])
        self.conv_branch = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', 
                                   activation='relu')
        ])
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        # Branch 1: Conv Block
        branch1_output = self.conv1_block(inputs)

        # Branch 2: Convolutional Layer
        branch2_output = self.conv_branch(inputs)

        # Add the two branches
        combined_output = tf.keras.layers.Add()( [branch1_output, branch2_output])

        # Flatten and Dense layer
        flattened_output = self.flatten(combined_output)
        output = self.dense(flattened_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Example input shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)