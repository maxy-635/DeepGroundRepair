import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.branch1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation=tf.keras.activations.relu
        )
        self.branch2 = tf.keras.layers.Sequential([
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='same',
                activation=tf.keras.activations.relu
            ),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                activation=tf.keras.activations.relu
            )
        ])
        self.branch3 = tf.keras.layers.Sequential([
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='same',
                activation=tf.keras.activations.relu
            ),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(5, 5),
                strides=(1, 1),
                padding='same',
                activation=tf.keras.activations.relu
            )
        ])
        self.branch4 = tf.keras.layers.Sequential([
            tf.keras.layers.MaxPool2D(
                pool_size=(3, 3),
                strides=(2, 2),
                padding='same'
            ),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='same',
                activation=tf.keras.activations.relu
            )
        ])
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            units=128,
            activation=tf.keras.activations.relu
        )
        self.dense2 = tf.keras.layers.Dense(
            units=10,
            activation=tf.keras.activations.softmax
        )

    def call(self, inputs):
        branch1 = self.branch1(inputs)
        branch2 = self.branch2(branch1)
        branch3 = self.branch3(branch2)
        branch4 = self.branch4(branch3)
        concatenated = tf.keras.layers.Concatenate(axis=-1)([branch1, branch2, branch3, branch4])
        flattened = self.flatten(concatenated)
        dense1 = self.dense1(flattened)
        output = self.dense2(dense1)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)