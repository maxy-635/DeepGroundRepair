import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First part: Feature Extraction
        self.conv1_1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.pool1 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid'
        )
        self.conv1_2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.pool1_2 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid'
        )

        # Second part: Enhancement
        self.conv2_1 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.dropout1 = tf.keras.layers.Dropout(
            rate=0.25
        )
        self.conv2_2 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )

        # Upsampling
        self.up1_1 = tf.keras.layers.Conv2DTranspose(
            filters=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation='relu'
        )
        self.skip_conv1_1 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.up2_1 = tf.keras.layers.Conv2DTranspose(
            filters=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation='relu'
        )
        self.skip_conv1_2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.up3_1 = tf.keras.layers.Conv2DTranspose(
            filters=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation='relu'
        )
        self.skip_conv2_1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )

        # Classification
        self.conv_final = tf.keras.layers.Conv2D(
            filters=10,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation=tf.keras.activations.softmax
        )

    def call(self, x):
        # First part
        x = self.conv1_1(x)
        x = self.pool1(x)
        x = self.conv1_2(x)
        x = self.pool1_2(x)

        # Second part
        x = self.conv2_1(x)
        x = self.dropout1(x)
        x = self.conv2_2(x)

        # Upsampling
        x = self.up1_1(x)
        x = tf.add(x, self.skip_conv1_1(x))
        x = self.up2_1(x)
        x = tf.add(x, self.skip_conv1_2(x))
        x = self.up3_1(x)
        x = tf.add(x, self.skip_conv2_1(x))

        # Classification
        x = self.conv_final(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)