import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: 3x3 convolutions
        self.branch1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        ])

        # Branch 2: 1x1 convolutions followed by two 3x3 convolutions
        self.branch2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        ])

        # Branch 3: Max pooling
        self.branch3 = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        ])

        # Concatenation layer
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        # Flatten layer
        self.flatten = tf.keras.layers.Flatten()

        # Fully connected layers
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Process inputs through each branch
        branch1_output = self.branch1(inputs)
        branch2_output = self.branch2(inputs)
        branch3_output = self.branch3(inputs)

        # Concatenate the outputs
        fused_features = self.concatenate([branch1_output, branch2_output, branch3_output])

        # Flatten and pass through fully connected layers
        flattened_features = self.flatten(fused_features)
        dense1_output = self.dense1(flattened_features)
        output = self.dense2(dense1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)