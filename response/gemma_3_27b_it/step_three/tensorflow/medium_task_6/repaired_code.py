import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolution layer
        self.conv_initial = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'
        )

        # Parallel blocks
        self.block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU()
        ])
        self.block2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU()
        ])
        self.block3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU()
        ])

        # Add layer
        self.add_layer = tf.keras.layers.Add()

        # Flatten and dense layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Initial convolution
        x = self.conv_initial(inputs)

        # Parallel blocks
        block1_output = self.block1(x)
        block2_output = self.block2(x)
        block3_output = self.block3(x)

        # Add outputs of blocks and initial convolution
        added_output = self.add_layer([x, block1_output, block2_output, block3_output])

        # Flatten and dense layers
        x = self.flatten(added_output)
        x = self.dense1(x)
        return self.dense2(x)


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor with shape (batch_size, height, width, channels)
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor)