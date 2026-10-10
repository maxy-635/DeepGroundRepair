import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.concat1 = tf.keras.layers.Concatenate(axis=-1)

        self.conv4 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.avgpool1 = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.dense1 = tf.keras.layers.Dense(
            units=128,
            activation='relu'
        )
        self.dense2 = tf.keras.layers.Dense(
            units=10,
            activation='softmax'
        )

    def call(self, inputs):
        # First block
        x = tf.keras.layers.Input(inputs.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.conv3(x)
        x = self.bn1(x)
        x = self.concat1([x, x, x])

        # Second block
        y = self.conv4(x)
        y = self.avgpool1(y)
        z = self.dense1(y)
        output = self.dense2(z)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)