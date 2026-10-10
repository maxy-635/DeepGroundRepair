import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.conv3 = tf.keras.layers.Conv2D(filters='mask', kernel_size='mask', strides='mask', padding='mask', activation='mask')






        self.maxpool = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            units=512,
            activation='relu'
        )
        self.dense2 = tf.keras.layers.Dense(
            units=10,
            activation='softmax'
        )

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        x = self.conv1(inputs)  # output tensor shape: (1, 32, 32, 32)
        x = self.maxpool(x)  # output tensor shape: (1, 16, 16, 32)
        x = self.conv2(x)  # output tensor shape: (1, 16, 16, 64)
        x = self.maxpool(x)  # output tensor shape: (1, 8, 8, 64)
        x = self.conv3(x)  # output tensor shape: (1, 8, 8, 128)

        path1 = x  # output tensor shape: (1, 8, 8, 128)
        path2 = self.conv3(inputs)

        x = tf.keras.layers.Add()([path1, path2])
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x


if __name__ == "__main__":
    # Load the CIFAR-10 dataset
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Preprocess the data
    #x_train = x_train.astype('float32') / 255.0
    #x_test = x_test.astype('float32') / 255.0

    # Create the DL_Model
    model = DL_Model()

    # Instantiate the model
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Batch size 1, 32x32 image, 3 color channels
    output_tensor = model(input_tensor)
    print(output_tensor.shape)