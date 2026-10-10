import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 convolution
        self.branch1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.Dropout(rate=0.2)
        ])

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.Dropout(rate=0.2)
        ])

        # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
        self.branch3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.Dropout(rate=0.2)
        ])

        # Branch 4: Average pooling followed by 1x1 convolution
        self.branch4 = tf.keras.Sequential([
            tf.keras.layers.AveragePooling2D(pool_size='mask', strides='mask', padding='mask'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.Dropout(rate=0.2)
        ])

        # Concatenation layer
        self.concat = tf.keras.layers.Concatenate(axis=-1)

        # Fully connected layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu)
        self.dense2 = tf.keras.layers.Dense(units=64, activation=tf.keras.activations.relu)
        self.dense3 = tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax)

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        branch1_output = self.branch1(inputs)  # output tensor shape: (1, 32, 32, 32)
        branch2_output = self.branch2(inputs)  # output tensor shape: (1, 32, 32, 32)
        branch3_output = self.branch3(inputs)  # output tensor shape: (1, 32, 32, 32)
        branch4_output = self.branch4(inputs)  # output tensor shape of this layer should be (1, 32, 32, 32)

        concatenated = self.concat([branch1_output, branch2_output, branch3_output, branch4_output])

        flattened = self.flatten(concatenated)
        dense1_output = self.dense1(flattened)
        dense2_output = self.dense2(dense1_output)
        output = self.dense3(dense2_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape: (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)