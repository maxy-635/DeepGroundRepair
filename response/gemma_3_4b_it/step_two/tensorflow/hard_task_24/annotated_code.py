import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same'
        )
        self.branch1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same'
            ),
            tf.keras.layers.MaxPool2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='valid'
            )
        ])
        self.branch2 = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='valid'
            ),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same'
            )
        ])
        self.branch3 = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='valid'
            ),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same'
            ),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size='mask', strides='mask', padding='mask')





        ])
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.conv_final = tf.keras.layers.Conv2D(
            filters=10,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same'
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            units=128,
            activation=tf.nn.relu
        )
        self.dense2 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu
        )
        self.dense3 = tf.keras.layers.Dense(
            units=10,
            activation='softmax'
        )

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        x = self.conv1(inputs)  # output tensor shape: (1, 32, 32, 32)
        x1 = self.branch1(x)  # output tensor shape: (1, 16, 16, 32)
        x2 = self.branch2(x)  # output tensor shape: (1, 16, 16, 64)
        x3 = self.branch3(x)  # output tensor shape of this layer should be (1, 16, 16, 64)
        x = self.concat([x1, x2, x3])
        x = self.conv_final(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

if __name__ == "__main__":
    # Load CIFAR-10 dataset
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    #x_train = x_train / 127.5 - 1.0
    #x_test = x_test / 127.5 - 1.0

    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)