import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters='mask', kernel_size='mask', strides='mask', padding='mask')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, padding='same')
        self.conv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='same')
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (32, 32, 3)
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)