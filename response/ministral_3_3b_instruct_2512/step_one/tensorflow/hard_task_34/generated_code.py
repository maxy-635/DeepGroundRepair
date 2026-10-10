import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.main_path = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None),
            tf.keras.layers.ReLU(),
            tf.keras.layers.SeparableConv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None),
            tf.keras.layers.Concatenate(axis=-1)
        ])

        self.branch_path = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        ])

        self.add_layer = tf.keras.layers.Add()
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layer = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        main_path_output = self.main_path(inputs)
        branch_path_output = self.branch_path(inputs)
        fused_output = self.add_layer([main_path_output, branch_path_output])
        flattened_output = self.flatten_layer(fused_output)
        output = self.dense_layer(flattened_output)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Example input shape for MNIST
    output_tensor = model(input_tensor)
    print(output_tensor.shape)