import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial Convolution Layer
        self.initial_conv = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )

        # Parallel Blocks
        self.block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                activation='relu'
            ),
            tf.keras.layers.BatchNormalization(axis=-1)
        ])
        self.block2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                activation='relu'
            ),
            tf.keras.layers.BatchNormalization(axis=-1)
        ])
        self.block3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                activation='relu'
            ),
            tf.keras.layers.BatchNormalization(axis=-1)
        ])

        # Add outputs of blocks to initial convolution
        self.add = tf.keras.layers.Add()

        # Flatten and Dense Layers
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
        x = self.initial_conv(inputs)
        x = self.add([x, self.block1(x), self.block2(x), self.block3(x)])
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


if __name__ == "__main__":
    # Load CIFAR-10 dataset
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Preprocess the data (e.g., normalize) - not required for the model definition itself
    #x_train = x_train.astype('float32') / 255.0
    #x_test = x_test.astype('float32') / 255.0

    # Instantiate the model
    model = DL_Model()

    # Dummy input tensor
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Batch size 1, 32x32 image, 3 color channels
    output_tensor = model(input_tensor)
    print(output_tensor.shape)