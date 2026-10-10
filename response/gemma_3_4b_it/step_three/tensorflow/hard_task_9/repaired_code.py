import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same')

        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        self.avg_pool1 = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)

        self.dense1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Branch 1
        branch1 = self.conv1(inputs)

        # Branch 2
        branch2 = self.conv2(inputs)

        # Branch 3
        branch3 = self.conv3(inputs)

        # Concatenate the branches
        merged = self.concatenate([branch1, branch2, branch3])

        # Add the original input to the merged output
        merged_with_input = tf.keras.layers.Add()( [merged, inputs])

        # Global Average Pooling
        pooled = self.avg_pool1(merged_with_input)

        # Dense layers
        dense1_output = self.dense1(pooled)
        output = self.dense2(dense1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)