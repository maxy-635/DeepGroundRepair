import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1
        self.path1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')

        # Path 2
        self.path2_conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.path2_conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 7), strides=1, padding='same', activation='relu')
        self.path2_conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 1), strides=1, padding='same', activation='relu')

        # Path 3
        self.path3_conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.path3_conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 7), strides=1, padding='same', activation='relu')
        self.path3_conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 1), strides=1, padding='same', activation='relu')
        self.path3_conv4 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 7), strides=1, padding='same', activation='relu')
        self.path3_conv5 = tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 1), strides=1, padding='same', activation='relu')

        # Path 4
        self.path4_pool = tf.keras.layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')
        self.path4_conv = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')

        # Concatenation
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        # Flatten and Dense
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Path 1
        path1 = self.path1(inputs)  # output tensor shape: (1, 32, 32, 32)

        # Path 2
        path2 = self.path2_conv1(inputs)  # output tensor shape: (1, 32, 32, 32)
        path2 = self.path2_conv2(path2)  # output tensor shape: (1, 32, 32, 32)
        path2 = self.path2_conv3(path2)  # output tensor shape: (1, 32, 32, 32)

        # Path 3
        path3 = self.path3_conv1(inputs)  # output tensor shape: (1, 32, 32, 32)
        path3 = self.path3_conv2(path3)  # output tensor shape: (1, 32, 32, 32)
        path3 = self.path3_conv3(path3)  # output tensor shape: (1, 32, 32, 32)
        path3 = self.path3_conv4(path3)  # output tensor shape: (1, 32, 32, 32)
        path3 = self.path3_conv5(path3)  # output tensor shape: (1, 32, 32, 32)

        # Path 4
        path4 = self.path4_pool(inputs)  # output tensor shape of this layer should be (1, 32, 32, 3)
        path4 = self.path4_conv(path4)

        # Concatenation
        merged = self.concatenate([path1, path2, path3, path4])

        # Flatten and Dense
        flattened = self.flatten(merged)
        output = self.dense(flattened)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # CIFAR-10 input shape: (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)