import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )
        self.dropout1 = tf.keras.layers.Dropout(
            rate=0.25
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.sep_conv1 = tf.keras.layers.SeparableConv2D(
            filters=64,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )
        self.dropout_sep1 = tf.keras.layers.Dropout(
            rate=0.25
        )
        self.sep_conv2 = tf.keras.layers.SeparableConv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )
        self.dropout_sep2 = tf.keras.layers.Dropout(
            rate=0.25
        )
        self.sep_conv3 = tf.keras.layers.SeparableConv2D(
            filters=64,
            kernel_size=5,
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )
        self.dropout_sep3 = tf.keras.layers.Dropout(
            rate=0.25
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(
            units=10,
            activation=tf.keras.activations.softmax
        )

    def call(self, inputs):
        # Main path
        x = self.conv1(inputs)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # Branch path
        y = inputs

        # Combined output
        x = tf.keras.layers.concatenate([x, y])

        # Separable convolutional blocks
        z1 = self.sep_conv1(x)
        z1 = self.dropout_sep1(z1)
        z2 = self.sep_conv2(z1)
        z2 = self.dropout_sep2(z2)
        z3 = self.sep_conv3(z2)
        z3 = self.dropout_sep3(z3)

        # Concatenate separable conv outputs
        x = tf.keras.layers.concatenate([x, z3])

        # Flatten and dense layer
        x = self.flatten(x)
        x = self.dense(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)