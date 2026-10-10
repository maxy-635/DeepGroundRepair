import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.input_layer = tf.keras.layers.Input(shape=(32, 32, 3))
        self.main_path = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(keepdims=True),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu')
        ])
        self.branch_path = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu')
        ])
        self.flatten = tf.keras.layers.Flatten()
        self.reshape = tf.keras.layers.Reshape((10,))
        self.multiply = tf.keras.layers.Multiply()
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.output_layer = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        main_output = self.main_path(inputs)  # output tensor shape: (1, 1, 1, 10)
        branch_output = self.branch_path(inputs)  # output tensor shape: (1, 32, 32, 128)

        reshaped_main = self.reshape(main_output)  # output tensor shape: (1, 10)
        reshaped_branch = branch_output  # output tensor shape: (1, 32, 32, 128)

        combined = self.multiply([inputs, reshaped_main])
        combined = self.concat([combined, reshaped_branch])
        output = self.output_layer(combined)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)