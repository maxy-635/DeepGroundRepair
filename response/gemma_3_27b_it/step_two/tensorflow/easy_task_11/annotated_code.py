import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.avg_pool = tf.keras.layers.AveragePooling2D(
            pool_size=(5, 5), strides=(3, 3), padding='valid'
        )
        self.conv1d = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.avg_pool(inputs)
        x = self.conv1d(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor)