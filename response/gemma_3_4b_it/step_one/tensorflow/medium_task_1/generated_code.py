import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Convolutional Layer 1
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        # Max Pooling Layer 1
        self.pool1 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same'
        )
        # Convolutional Layer 2
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        # Max Pooling Layer 2
        self.pool2 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same'
        )
        # Flatten Layer
        self.flatten = tf.keras.layers.Flatten()
        # Dense Layer 1
        self.dense1 = tf.keras.layers.Dense(
            units=128,
            activation='relu'
        )
        # Dense Layer 2 (Output Layer)
        self.dense2 = tf.keras.layers.Dense(
            units=10,
            activation='softmax'
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Batch size 1, 32x32 image, 3 color channels
    output_tensor = model(input_tensor)
    print(output_tensor.shape)