import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.bn3 = tf.keras.layers.BatchNormalization(axis=-1)
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu)
        self.dense2 = tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.keras.activations.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.keras.activations.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = tf.keras.activations.relu(x)

        x = self.concat([x, x, x])
        x = self.flatten(x)

        x = self.dense1(x)
        x = self.dense2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape: (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)