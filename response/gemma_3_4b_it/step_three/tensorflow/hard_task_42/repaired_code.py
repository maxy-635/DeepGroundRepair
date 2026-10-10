import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=1, strides=1, padding='same'
        )
        self.maxpool1_1 = tf.keras.layers.MaxPool2D(
            pool_size=1, strides=1, padding='same'
        )
        self.maxpool1_2 = tf.keras.layers.MaxPool2D(
            pool_size=2, strides=2, padding='same'
        )
        self.maxpool1_3 = tf.keras.layers.MaxPool2D(
            pool_size=4, strides=4, padding='same'
        )

        self.dropout1 = tf.keras.layers.Dropout(rate=0.25)

        # Flatten and regularize
        self.flatten1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
        self.dropout2 = tf.keras.layers.Dropout(rate=0.5)

        # Block 2
        self.conv2_1 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=1, strides=1, padding='same'
        )
        self.avgpool2_1 = tf.keras.layers.AvgPool2D(
            pool_size=1, strides=1, padding='same'
        )
        self.conv2_2 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=1, strides=1, padding='same'
        )
        self.conv2_3 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=7, strides=1, padding='same'
        )
        self.conv2_4 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=1, strides=1, padding='same'
        )
        self.conv2_5 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=1, strides=1, padding='same'
        )

        self.concat2 = tf.keras.layers.Concatenate(axis=None)

        # Final classification layers
        self.dense3 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        # Block 1
        x = self.conv1(inputs)  # output tensor shape: (1, 28, 28, 32)
        x = self.maxpool1_1(x)  # output tensor shape: (1, 28, 28, 32)
        x = self.maxpool1_2(x)  # output tensor shape: (1, 14, 14, 32)
        x = self.maxpool1_3(x)  # output tensor shape: (1, 4, 4, 32)
        x = self.dropout1(x)  # output tensor shape: (1, 4, 4, 32)

        # Flatten and regularize
        x = self.flatten1(x)  # output tensor shape: (1, 512)
        x = self.dense1(x)  # output tensor shape: (1, 128)
        x = self.dropout2(x)  # output tensor shape: (1, 128)

        # Block 2
        x_concat = tf.concat([  # output tensor shape: (1, 28, 28, 321)
            self.conv2_1(inputs),
            self.avgpool2_1(inputs),
            self.conv2_2(inputs),
            self.conv2_3(inputs),
            self.conv2_4(inputs),
            self.conv2_5(inputs)
        ], axis=-1)
        x = self.concat2(x_concat)

        # Final classification
        x = self.dense3(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Example input shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)