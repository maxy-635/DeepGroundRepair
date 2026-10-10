import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main pathway
        self.conv1x1_main = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')

        # Parallel branch
        self.conv_parallel_1x1 = tf.keras.layers.Conv2D(filters=64, kernel_size='mask', strides='mask', padding='mask')
        self.conv_parallel_1x3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='valid')
        self.conv_parallel_3x1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='valid')

        # Concatenation and final 1x1 convolution
        self.concat = tf.keras.layers.Concatenate(axis=3)
        self.conv_final = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')

        # Direct connection
        self.add = tf.keras.layers.Add()

        # Fully connected layers for classification
        self.dense1 = tf.keras.layers.Dense(units=256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Main pathway
        x_main = self.conv1x1_main(inputs)  # output tensor shape: (1, 32, 32, 64)

        # Parallel branch
        x_parallel_1x1 = self.conv_parallel_1x1(inputs)  # output tensor shape of this layer should be (1, 32, 30, 64)
        x_parallel_1x3 = self.conv_parallel_1x3(inputs)  # output tensor shape: (1, 32, 30, 64)
        x_parallel_3x1 = self.conv_parallel_3x1(inputs)  # output tensor shape: (1, 30, 32, 64)

        # Concatenate outputs along channel dimension
        x_concat = self.concat([x_parallel_1x1, x_parallel_1x3, x_parallel_3x1])

        # Add main pathway and concatenated branch
        x = self.add([x_main, x_concat])

        # Final 1x1 convolution
        x = self.conv_final(x)

        # Flatten and pass through dense layers for classification
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    dummy_input = tf.random.normal(shape=(1, 32, 32, 3))

    model = DL_Model()
    output = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)