import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1
        branch1 = tf.keras.Sequential([  # output tensor shape: (unknown)
            tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Dropout(rate=0.2)
        ])

        # Branch 2
        branch2 = tf.keras.Sequential([  # output tensor shape: (unknown)
            tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=7, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=7, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Dropout(rate=0.2)
        ])

        # Branch 3
        branch3 = tf.keras.Sequential([  # output tensor shape: (unknown)
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            tf.keras.layers.Dropout(rate=0.2)
        ])

        # Concatenate branches
        self.concatenate = tf.keras.layers.Concatenate(axis='mask')

        # Fully connected layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=10, activation='softmax')

        self.branch1 = branch1
        self.branch2 = branch2
        self.branch3 = branch3

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        output1 = self.branch1(inputs)  # output tensor shape: (1, 32, 32, 32)
        output2 = self.branch2(inputs)  # output tensor shape: (1, 32, 32, 32)
        output3 = self.branch3(inputs)  # output tensor shape: (1, 16, 16, 3)

        concatenated = self.concatenate([output1, output2, output3])

        flattened = self.flatten(concatenated)
        dense1_output = self.dense1(flattened)
        dense2_output = self.dense2(dense1_output)
        output = self.dense3(dense2_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)