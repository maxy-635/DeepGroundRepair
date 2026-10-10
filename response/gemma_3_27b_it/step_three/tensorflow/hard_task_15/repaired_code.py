import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=False)
        self.dense1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=32, activation='relu')
        self.reshape = tf.keras.layers.Reshape(target_shape=(1, 32))
        self.multiply = tf.keras.layers.Multiply()

        # Branch Path
        # No modification in branch path

        # Combined Path
        self.flatten = tf.keras.layers.Flatten()
        self.dense3 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense4 = tf.keras.layers.Dense(units=10, activation='softmax')
        self.add = tf.keras.layers.Add()

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Main Path
        x = self.conv1(inputs)  # output tensor shape: (1, 32, 32, 32)
        x = self.conv2(x)  # output tensor shape: (1, 32, 32, 64)
        pooled_x = self.global_avg_pool(x)  # output tensor shape: (1, 64)
        dense1_x = self.dense1(pooled_x)  # output tensor shape: (1, 64)
        dense2_x = self.dense2(dense1_x)  # output tensor shape: (1, 32)
        reshaped_x = self.reshape(dense2_x)
        weighted_x = self.multiply([inputs, reshaped_x])

        # Branch Path
        branch_x = inputs

        # Combined Path
        combined_x = self.add([weighted_x, branch_x])
        flattened_x = self.flatten(combined_x)
        dense3_x = self.dense3(flattened_x)
        output = self.dense4(dense3_x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)