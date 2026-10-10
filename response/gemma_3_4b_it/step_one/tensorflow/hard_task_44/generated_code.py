import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same')
        self.conv1_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')
        self.conv1_5 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same')
        self.dropout1 = tf.keras.layers.Dropout(rate=0.25)
        self.conv2_1x1 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')
        self.conv2_<1x1> = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')
        self.conv2_3x3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.conv2_<1x1>_5x5 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')
        self.maxpool2d = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1_1(inputs)
        x = self.conv1_3(x)
        x = self.conv1_5(x)
        x = self.dropout1(x)
        x = self.conv2_1x1(x)
        x = self.conv2_<1x1>(x)
        x = self.conv2_3x3(x)
        x = self.conv2_<1x1>_5x5(x)
        x = self.maxpool2d(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)