import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First convolutional path
        self.conv1_1 = tf.keras.layers.Conv2D(filters='mask', kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size='mask', strides='mask', padding='mask')

        # Second convolutional path
        self.conv2_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Flatten and Dense layer
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        # First path
        x1 = self.conv1_1(inputs)  # output tensor shape of this layer should be (1, 28, 28, 1)
        x1 = self.maxpool1(x1)  # output tensor shape of this layer should be (1, 28, 28, 32)

        # Second path
        x2 = self.conv2_1(inputs)  # output tensor shape: (1, 28, 28, 64)
        x2 = self.maxpool2(x2)  # output tensor shape: (1, 14, 14, 64)
        
        # Add the outputs of both paths to the input
        x = tf.keras.layers.Add()([inputs, x1, x2])

        # Flatten and classify
        x = self.flatten(x)
        return self.dense(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Dummy input tensor with shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor)