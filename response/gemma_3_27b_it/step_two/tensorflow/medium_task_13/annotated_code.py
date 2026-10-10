import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the layers
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.keras.activations.relu)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv2 = tf.keras.layers.Conv2D(filters='mask', kernel_size='mask', strides='mask', padding='mask', activation='mask')
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.keras.activations.relu)
        self.bn3 = tf.keras.layers.BatchNormalization(axis=-1)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu)
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')  # 10 classes for CIFAR-10

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Concatenation of input and output from each convolutional layer
        x = self.conv1(inputs)  # output tensor shape: (1, 32, 32, 32)
        x = self.bn1(x)  # output tensor shape: (1, 32, 32, 32)
        x = tf.keras.layers.Concatenate(axis=-1)([inputs, x])  # output tensor shape: (1, 32, 32, 35)

        x = self.conv2(x)  # output tensor shape: (1, 32, 32, 64)
        x = self.bn2(x)  # output tensor shape: (1, 32, 32, 64)
        x = tf.keras.layers.Concatenate(axis=-1)([x, self.conv1(inputs)])  # output tensor shape: (1, 32, 32, 96)

        x = self.conv3(x)  # output tensor shape: (1, 32, 32, 128)
        x = self.bn3(x)  # output tensor shape: (1, 32, 32, 128)
        x = tf.keras.layers.Concatenate(axis=-1)([x, self.conv2(x)])

        # Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 color channels (RGB)
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output: (1, 10)