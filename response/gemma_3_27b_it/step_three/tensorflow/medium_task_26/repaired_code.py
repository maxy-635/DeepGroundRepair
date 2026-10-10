import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.conv1x1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.conv3x3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')  # Assuming 10 classes

    def call(self, inputs):
        x = self.conv1(inputs)
        x1x1 = self.conv1x1(x)
        x3x3 = self.conv3x3(x)
        x = self.concatenate([x1x1, x3x3])
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 64))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)