import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1
        self.conv1_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.bn1_1 = tf.keras.layers.BatchNormalization(axis=-1)

        # Block 2
        self.conv2_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.bn2_1 = tf.keras.layers.BatchNormalization(axis=-1)

        # Block 3
        self.conv3_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.bn3_1 = tf.keras.layers.BatchNormalization(axis=-1)

        # Flatten and Dense layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1
        x1 = self.conv1_1(inputs)
        x1 = self.bn1_1(x1)

        # Block 2
        x2 = self.conv2_1(inputs)
        x2 = self.bn2_1(x2)
        x = tf.keras.layers.Concatenate(axis=-1)([x1, x2])

        # Block 3
        x3 = self.conv3_1(x)
        x3 = self.bn3_1(x3)
        x = tf.keras.layers.Concatenate(axis=-1)([x, x3])

        # Flatten and Dense layers
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)