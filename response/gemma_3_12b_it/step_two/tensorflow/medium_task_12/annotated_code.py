import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        self.bn1_1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu1_1 = tf.keras.layers.ReLU()

        # Block 2
        self.conv2_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        self.bn2_1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu2_1 = tf.keras.layers.ReLU()

        # Block 3
        self.conv3_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        self.bn3_1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu3_1 = tf.keras.layers.ReLU()

        # Fully Connected Layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=512, activation=None)
        self.relu_dense1 = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1
        x = self.conv1_1(inputs)
        x = self.bn1_1(x)
        x = self.relu1_1(x)

        # Block 2
        y = self.conv2_1(x)
        y = self.bn2_1(y)
        y = self.relu2_1(y)
        y = tf.keras.layers.Concatenate(axis=-1)([x, y])

        # Block 3
        z = self.conv3_1(y)
        z = self.bn3_1(z)
        z = self.relu3_1(z)
        z = tf.keras.layers.Concatenate(axis=-1)([y, z])

        # Fully Connected Layers
        x = self.flatten(z)
        x = self.dense1(x)
        x = self.relu_dense1(x)
        x = self.dense2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)