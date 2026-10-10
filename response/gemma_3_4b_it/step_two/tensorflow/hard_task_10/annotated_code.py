import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same')
        self.concatenate = tf.keras.layers.concatenate(inputs='mask', axis='mask')
        self.conv4 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.global_average_pooling2d = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        path1 = self.conv1(inputs)
        path2 = self.conv2(inputs)
        path3 = self.conv3(inputs)
        merged = tf.keras.layers.add([path1, path2, path3])
        merged = self.conv4(merged)
        flattened = self.global_average_pooling2d(merged)
        output = self.dense1(flattened)
        output = self.dense2(output)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)