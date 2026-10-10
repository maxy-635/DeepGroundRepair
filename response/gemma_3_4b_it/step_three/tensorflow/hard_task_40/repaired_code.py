import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.avg_pool1 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='VALID')
        self.avg_pool2 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='VALID')
        self.avg_pool3 = tf.keras.layers.AveragePooling1D(pool_size=4, strides=4, padding='VALID')

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.reshape = tf.keras.layers.Reshape((128, 4, 4))

        self.conv1_path1 = tf.keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, padding='VALID')
        self.dropout1 = tf.keras.layers.Dropout(rate=0.25)
        self.conv2_path2 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='VALID')
        self.conv3_path2 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='VALID')
        self.dropout2 = tf.keras.layers.Dropout(rate=0.25)
        self.conv4_path3 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='VALID')
        self.dropout3 = tf.keras.layers.Dropout(rate=0.25)
        self.avg_pool_path4 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=1, padding='VALID')
        self.conv5_path4 = tf.keras.layers.Conv1D(filters=128, kernel_size=1, strides=1, padding='VALID')
        self.dropout4 = tf.keras.layers.Dropout(rate=0.25)

        self.concatenate = tf.keras.layers.Concatenate(axis=1)
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        x = self.avg_pool1(inputs)
        x = self.avg_pool2(x)
        x = self.avg_pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.reshape(x)

        path1 = self.conv1_path1(x)
        path1 = self.dropout1(path1)

        path2 = tf.keras.layers.concatenate([self.conv2_path2(x), self.conv3_path2(x)])
        path2 = self.dropout2(path2)

        path3 = self.conv4_path3(x)
        path3 = self.dropout3(path3)

        path4 = self.avg_pool_path4(x)
        path4 = self.conv5_path4(path4)
        path4 = self.dropout4(path4)

        x = self.concatenate([path1, path2, path3, path4])
        x = self.dense2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)