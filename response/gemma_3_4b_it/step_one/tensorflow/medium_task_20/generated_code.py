import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.conv4 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        path1 = self.conv1(inputs)
        path2 = tf.keras.layers.Concatenate(axis=-1)([self.conv2(inputs), self.conv3(inputs)])
        path3 = self.conv3(inputs)
        path4 = self.conv4(inputs)

        merged = tf.keras.layers.Concatenate(axis=-1)([path1, path2, path3, path4])
        flattened = self.flatten(merged)
        dense_output = self.dense1(flattened)
        output = self.output_layer(dense_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)