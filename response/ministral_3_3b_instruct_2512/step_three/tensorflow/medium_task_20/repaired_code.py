import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1x1 = tf.keras.layers.Conv2D(64, kernel_size=(1, 1), activation='relu')
        self.conv3x3_1 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')
        self.conv3x3_2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')
        self.conv3x3 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')
        self.dense = tf.keras.layers.Dense(128, activation='softmax')

    def call(self, inputs):
        path1 = self.conv1x1(inputs)
        path2 = self.conv3x3_1(tf.keras.layers.Conv2D(64, kernel_size=(1, 1), activation='relu')(inputs))
        path2 = self.conv3x3_2(path2)
        path3 = self.conv3x3(tf.keras.layers.Conv2D(64, kernel_size=(1, 1), activation='relu')(inputs))
        path4 = self.maxpool(inputs)
        path4 = self.conv1x1(path4)
        concatenated = tf.keras.layers.concatenate([path1, path2, path3, path4], axis=-1)
        flattened = tf.keras.layers.Flatten()(concatenated)
        output = self.dense(flattened)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)