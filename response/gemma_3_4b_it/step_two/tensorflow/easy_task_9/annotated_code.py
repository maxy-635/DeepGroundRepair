import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.input_layer = tf.keras.layers.Input(shape=(28, 28, 1))

        # 1x1 convolutional layer
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same')
        self.add = tf.keras.layers.Add()

        # 3x3 depthwise separable convolutional layer
        self.conv2 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1)
        self.conv2_activation = tf.keras.layers.Activation('relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same')

        # 1x1 convolutional layer to reduce dimensionality
        self.conv4 = tf.keras.layers.Conv2D(filters=16, kernel_size=1, strides=1, padding='same')

        # Flatten layer
        self.flatten = tf.keras.layers.Flatten()

        # Fully connected layer
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.add([inputs, x])
        x = self.conv2_activation(self.conv2(x))
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)