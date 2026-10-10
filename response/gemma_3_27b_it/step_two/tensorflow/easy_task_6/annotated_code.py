import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'
        )
        # Branch path
        # Combination
        self.add_layer = tf.keras.layers.Add()
        # Classification
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main path
        x = self.conv1(inputs)
        x = self.conv2(x)
        # Branch path
        branch = inputs
        # Combination
        x = self.add_layer([x, branch])
        # Classification
        x = self.flatten(x)
        return self.dense(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor)