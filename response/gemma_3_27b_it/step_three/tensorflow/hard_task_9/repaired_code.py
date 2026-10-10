import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Feature extraction branches
        self.branch1 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')
        self.branch2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')
        ])
        self.branch3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')
        ])

        # Concatenation and dimension adjustment
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.conv_concat = tf.keras.layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same', activation='relu')

        # Residual connection
        self.add = tf.keras.layers.Add()

        # Classification layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Feature extraction branches
        branch1_output = self.branch1(inputs)
        branch2_output = self.branch2(inputs)
        branch3_output = self.branch3(inputs)

        # Concatenation and dimension adjustment
        concat_output = self.concat([branch1_output, branch2_output, branch3_output])
        concat_output = self.conv_concat(concat_output)

        # Residual connection
        output = self.add([inputs, concat_output])

        # Classification layers
        flattened_output = self.flatten(output)
        dense1_output = self.dense1(flattened_output)
        dense2_output = self.dense2(dense1_output)
        output = self.dense3(dense2_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)