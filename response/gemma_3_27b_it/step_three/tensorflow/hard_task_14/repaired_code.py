import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main Path
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=False)
        self.dense1 = tf.keras.layers.Dense(units=32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=32, activation='relu')
        self.reshape = tf.keras.layers.Reshape(target_shape=(1, 1, 32))  # Assuming 32 channels in input

        # Branch Path
        self.conv = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='same', activation='relu')

        # Combination and Classification
        self.add = tf.keras.layers.Add()
        self.dense3 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense4 = tf.keras.layers.Dense(units=32, activation='relu')
        self.dense5 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Main Path
        main_path = self.global_avg_pool(inputs)  # output tensor shape: (1, 3)
        main_path = self.dense1(main_path)  # output tensor shape: (1, 32)
        main_path = self.dense2(main_path)  # output tensor shape: (1, 32)
        main_path = self.reshape(main_path)  # output tensor shape: (1, 1, 1, 32)

        # Branch Path
        branch_path = self.conv(inputs)  # output tensor shape of this layer should be (1, 32, 32, 3)

        # Combination
        combined = self.add([inputs, branch_path])

        # Classification
        x = self.dense3(combined)
        x = tf.keras.layers.Flatten()(x)
        x = self.dense4(x)
        output = self.dense5(x)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)