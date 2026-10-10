import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the four parallel convolutional paths
        self.path1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        ])

        self.path2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        ])

        self.path3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        ])

        self.path4 = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='same'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        ])

        # Concatenate the outputs of the four paths
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        # Flatten the concatenated output
        self.flatten = tf.keras.layers.Flatten()

        # Dense layer
        self.dense = tf.keras.layers.Dense(units=128, activation='relu')

        # Output layer
        self.output_layer = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Pass the input through the four parallel paths
        path1_output = self.path1(inputs)  # output tensor shape: (1, 32, 32, 32)
        path2_output = self.path2(inputs)  # output tensor shape: (1, 32, 32, 32)
        path3_output = self.path3(inputs)  # output tensor shape: (1, 32, 32, 32)
        path4_output = self.path4(inputs)  # output tensor shape of this layer should be (1, 32, 32, 32)

        # Concatenate the outputs
        concatenated_output = self.concatenate([path1_output, path2_output, path3_output, path4_output])

        # Flatten the concatenated output
        flattened_output = self.flatten(concatenated_output)

        # Pass through the dense layer
        dense_output = self.dense(flattened_output)

        # Pass through the output layer
        output = self.output_layer(dense_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input tensor with shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)