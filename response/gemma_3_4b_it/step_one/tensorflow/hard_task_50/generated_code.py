import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.first_block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=1x1, strides=1, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=2x2, strides=2, padding='valid'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=2x2, strides=1, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=2x2, strides=2, padding='valid'),
            tf.keras.layers.Conv2D(filters=128, kernel_size=4x4, strides=1, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=2x2, strides=2, padding='valid'),
            tf.keras.layers.Dropout(rate=0.2),
        ])
        self.second_block = tf.keras.Sequential([
            tf.keras.layers.Reshape((1, 1, 128)),
            tf.keras.layers.Conv2D(filters=128, kernel_size=1x1, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=128, kernel_size=3x3, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=128, kernel_size=5x5, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=128, kernel_size=7x7, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Add(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    def call(self, inputs):
        x = self.first_block(inputs)
        x = self.second_block(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)