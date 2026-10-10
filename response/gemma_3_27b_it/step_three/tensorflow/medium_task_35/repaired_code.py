import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Stage 1: Downsampling
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Stage 2: Downsampling
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Additional Convolutional Layers
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(rate=0.25)

        # Upsampling and Skip Connections
        self.upsample1 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        x = self.add1([x, self.conv2(x)])

        self.upsample2 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.conv5 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.add2 = tf.keras.layers.Add()

        # Output Layer
        self.conv6 = tf.keras.layers.Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='softmax')

    def call(self, x):
        # the initial shape of x is: (1, 32, 32, 3)
        # Downsampling
        x = self.conv1(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.maxpool1(x)  # output tensor shape: (1, 16, 16, 32)
        x = self.conv2(x)  # output tensor shape: (1, 16, 16, 64)
        x = self.maxpool2(x)  # output tensor shape: (1, 8, 8, 64)

        # Additional Convolutional Layers
        x = self.conv3(x)  # output tensor shape: (1, 8, 8, 128)
        x = self.dropout1(x)  # output tensor shape: (1, 8, 8, 128)

        # Upsampling and Skip Connections
        x = self.upsample1(x)  # output tensor shape: (1, 16, 16, 128)
        x = self.conv4(x)  # output tensor shape: (1, 16, 16, 64)
        x = self.add1([x, self.conv2])

        x = self.upsample2(x)
        x = self.conv5(x)
        x = self.add2([x, self.conv1])

        # Output Layer
        x = self.conv6(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Dummy input tensor with shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)