import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=1,
            strides=1,
            padding='same',
            activation='relu'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu'
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=5,
            strides=1,
            padding='same',
            activation='relu'
        )
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv_branch = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=1,
            strides=1,
            padding='same',
            activation='relu'
        )
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
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.conv3(x)
        x = self.bn1(x)
        main_path = self.concat([x, x, x])

        # Branch path
        x = self.conv_branch(inputs)
        x = self.bn1(x)
        x = self.flatten(x)
        x = self.dense1(x)

        # Fuse paths
        combined = self.add([main_path, x])

        # Classification
        output = self.dense2(combined)

        return output


if __name__ == "__main__":
    # Load CIFAR-10 dataset
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Preprocess data (e.g., normalize)
    #x_train = x_train.astype('float32') / 255.0
    #x_test = x_test.astype('float32') / 255.0

    # Instantiate the model
    model = DL_Model()

    # Dummy input tensor
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Batch size 1, 32x32 image, 3 color channels
    output_tensor = model(input_tensor)
    print(output_tensor.shape)