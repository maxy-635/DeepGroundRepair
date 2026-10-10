import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.main_path = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=7,
                strides=1,
                padding='same',
                activation='relu'
            ),
            tf.keras.layers.LayerNormalization(axis=1, epsilon=1e-6),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=1,
                strides=1,
                padding='same',
                activation='relu'
            ),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=1,
                strides=1,
                padding='same',
                activation='relu'
            )
        ])
        self.branch_path = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu'
        )
        self.merge = tf.keras.layers.Add()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            units=128,
            activation='relu'
        )
        self.dense2 = tf.keras.layers.Dense(
            units=10,
            activation='softmax'
        )

    def call(self, inputs):
        main_path_output = self.main_path(inputs)
        branch_path_output = self.branch_path(inputs)
        combined_output = self.merge([main_path_output, branch_path_output])
        flattened_output = self.flatten(combined_output)
        dense1_output = self.dense1(flattened_output)
        output = self.dense2(dense1_output)
        return output


if __name__ == "__main__":
    # Load the CIFAR-10 dataset
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Preprocess the data (example - you might need more preprocessing)
    #x_train = x_train.astype('float32') / 255.0
    #x_test = x_test.astype('float32') / 255.0

    # Instantiate the DL_Model
    model = DL_Model()

    # Create a dummy input tensor
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Batch size 1, 32x32 image, 3 channels
    
    # Pass the input tensor through the model
    output_tensor = model(input_tensor)

    # Print the output shape
    print(output_tensor.shape)