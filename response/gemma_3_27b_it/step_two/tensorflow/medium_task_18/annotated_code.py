import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1x1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5 = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        x1 = self.conv1x1(inputs)
        x2 = self.conv3x3(inputs)
        x3 = self.conv5x5(inputs)
        x1 = self.maxpool(x1)
        x2 = self.maxpool(x2)
        x3 = self.maxpool(x3)
        x = self.concatenate([x1, x2, x3])
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)