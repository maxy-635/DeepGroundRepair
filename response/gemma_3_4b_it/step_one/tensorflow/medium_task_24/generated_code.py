import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, padding='same')
        self.conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=7, strides=1, padding='same')
        self.conv5 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.dropout1 = tf.keras.layers.Dropout(rate=0.25)
        self.dropout2 = tf.keras.layers.Dropout(rate=0.5)

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=256, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.dropout2(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)