import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(rate=0.25)
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(rate=0.25)
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch path
        self.branch = tf.keras.layers.Identity()

        # Combine paths
        self.add = tf.keras.layers.Add()

        # Classification layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main path
        x = self.conv1(inputs)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.conv3(x)

        # Branch path
        y = self.branch(inputs)

        # Combine paths
        combined = self.add([x, y])

        # Classification layers
        flattened = self.flatten(combined)
        output = self.dense(flattened)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)