import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.flatten1 = tf.keras.layers.Flatten()

        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.flatten2 = tf.keras.layers.Flatten()

        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
        self.flatten3 = tf.keras.layers.Flatten()

        self.concatenate = tf.keras.layers.Concatenate(axis=1)

        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        x1 = self.conv1(inputs)
        x1 = self.pool1(x1)
        x1 = self.flatten1(x1)

        x2 = self.conv2(inputs)
        x2 = self.pool2(x2)
        x2 = self.flatten2(x2)

        x3 = self.conv3(inputs)
        x3 = self.pool3(x3)
        x3 = self.flatten3(x3)

        x = self.concatenate([x1, x2, x3])
        x = self.dense1(x)
        return self.dense2(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)