import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same')
        self.avg_pool1 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)
        self.conv2_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=1, strides=1, padding='same')
        self.conv2_3 = tf.keras.layers.Conv2D(filters=16, kernel_size=1, strides=1, padding='same')
        self.concat2 = tf.keras.layers.Concatenate(axis=-1)
        self.conv3_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same')
        self.conv3_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same')
        self.conv3_4 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')
        self.conv3_5 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same')
        self.concat3 = tf.keras.layers.Concatenate(axis=-1)
        self.conv4_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')
        self.conv4_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')
        self.conv4_4 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.conv4_5 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')
        self.concat4 = tf.keras.layers.Concatenate(axis=-1)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        path1 = self.conv1(inputs)
        path2 = self.avg_pool1(inputs)
        path2 = self.conv2_1(path2)
        path2 = self.conv2_3(path2)
        path2 = self.concat2([path1, path2])

        path3 = self.conv3_1(inputs)
        path3 = self.conv3_3(path3)
        path3 = self.conv3_4(path3)
        path3 = self.conv3_5(path3)
        path3 = self.concat3([path1, path3])

        path4 = self.conv4_1(inputs)
        path4 = self.conv4_3(path4)
        path4 = self.conv4_4(path4)
        path4 = self.conv4_5(path4)
        path4 = self.concat4([path1, path4])

        x = self.flatten(path4)
        x = self.dense(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)