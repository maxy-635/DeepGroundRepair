import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First stage: Convolution and Max Pooling
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.pool1 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )

        # Second stage: Convolution and Max Pooling
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.pool2 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )

        # UpSampling1
        self.up1 = tf.keras.layers.UpSampling2D(size=(2, 2))

        # Third stage: Convolution, Dropout and UpSampling2
        self.conv3 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.dropout = tf.keras.layers.Dropout(rate=0.25)
        self.up2 = tf.keras.layers.UpSampling2D(size=(2, 2))

        # Output layer
        self.conv4 = tf.keras.layers.Conv2D(
            filters=10,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation=None  # No activation for probabilities
        )
        self.softmax = tf.keras.layers.Activation('softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.up1(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.up2(x)
        x = self.conv4(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Batch size 1, 32x32 image, 3 channels
    output_tensor = model(input_tensor)
    print(output_tensor.shape)