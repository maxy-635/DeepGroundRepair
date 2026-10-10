import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Feature Extraction
        self.conv1_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.conv2_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.conv3_1 = tf.keras.layers.Conv2D(filters='mask', kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.maxpool3 = tf.keras.layers.MaxPool2D(pool_size='mask', strides='mask', padding='mask')

        # Generalization Enhancement
        self.conv4_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(rate=0.25)
        self.conv4_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Upsampling with Skip Connections
        self.conv5_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.transpose_conv1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')
        self.add1 = tf.keras.layers.Add()

        self.conv6_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.transpose_conv2 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')
        self.add2 = tf.keras.layers.Add()

        self.conv7_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.transpose_conv3 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')
        self.add3 = tf.keras.layers.Add()

        # Output Layer
        self.conv8_1 = tf.keras.layers.Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation=None)

    def call(self, x):
        # the initial shape of x is: (1, 32, 32, 3)
        # Feature Extraction
        x = self.conv1_1(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.maxpool1(x)  # output tensor shape: (1, 16, 16, 32)
        x_1 = x  # output tensor shape: (1, 16, 16, 32)
        x = self.conv2_1(x)  # output tensor shape: (1, 16, 16, 64)
        x = self.maxpool2(x)  # output tensor shape: (1, 8, 8, 64)
        x_2 = x  # output tensor shape: (1, 8, 8, 64)
        x = self.conv3_1(x)  # output tensor shape of this layer should be (1, 8, 8, 64)
        x = self.maxpool3(x)  # output tensor shape of this layer should be (1, 8, 8, 128)
        x_3 = x

        # Generalization Enhancement
        x = self.conv4_1(x)  # output tensor shape: (1, 4, 4, 128)
        x = self.dropout1(x)  # output tensor shape: (1, 4, 4, 128)
        x = self.conv4_2(x)  # output tensor shape: (1, 4, 4, 128)

        # Upsampling with Skip Connections
        x = self.transpose_conv1(x)  # output tensor shape: (1, 8, 8, 64)
        x = self.add1([x, x_3])
        x = self.conv5_1(x)

        x = self.transpose_conv2(x)
        x = self.add2([x, x_2])
        x = self.conv6_1(x)

        x = self.transpose_conv3(x)
        x = self.add3([x, x_1])
        x = self.conv7_1(x)

        # Output Layer
        x = self.conv8_1(x)
        x = tf.nn.softmax(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)