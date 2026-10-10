import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First sequential feature extraction
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.pool1 = tf.keras.layers.AveragePooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same'
        )

        # Second sequential feature extraction
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.pool2 = tf.keras.layers.AveragePooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same'
        )

        # Additional convolutional layers
        self.conv3 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.conv4 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.conv5 = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )

        # Another average pooling layer
        self.pool3 = tf.keras.layers.AveragePooling2D(
            pool_size=(8, 8),
            strides=(8, 8),
            padding='same'
        )

        # Flatten the feature maps
        self.flatten = tf.keras.layers.Flatten()

        # Fully connected layers with dropout
        self.dense1 = tf.keras.layers.Dense(
            units=512,
            activation='relu'
        )
        self.dropout1 = tf.keras.layers.Dropout(
            rate=0.5
        )
        self.dense2 = tf.keras.layers.Dense(
            units=1000,
            activation='softmax'
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 224, 224, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)