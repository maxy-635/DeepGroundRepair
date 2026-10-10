import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1
        self.path1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')

        # Path 2
        self.path2_pool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')
        self.path2_conv = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')

        # Path 3
        self.path3_conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.path3_conv1x3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 3), strides=1, padding='same', activation='relu')
        self.path3_conv3x1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 1), strides=1, padding='same', activation='relu')

        # Path 4
        self.path4_conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.path4_conv3x3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.path4_conv1x3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 3), strides=1, padding='same', activation='relu')
        self.path4_conv3x1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 1), strides=1, padding='same', activation='relu')

        # Concatenation
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        # Flatten and Dense layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Path 1
        path1 = self.path1(inputs)  # output tensor shape: (1, 32, 32, 32)

        # Path 2
        path2 = self.path2_pool(inputs)  # output tensor shape of this layer should be (1, 32, 32, 3)
        path2 = self.path2_conv(path2)

        # Path 3
        path3 = self.path3_conv1(inputs)  # output tensor shape: (1, 32, 32, 32)
        path3_1x3 = self.path3_conv1x3(path3)  # output tensor shape: (1, 32, 32, 32)
        path3_3x1 = self.path3_conv3x1(path3)  # output tensor shape: (1, 32, 32, 32)
        path3 = self.concatenate([path3_1x3, path3_3x1])  # output tensor shape: (1, 32, 32, 64)

        # Path 4
        path4 = self.path4_conv1(inputs)  # output tensor shape: (1, 32, 32, 32)
        path4 = self.path4_conv3x3(path4)  # output tensor shape: (1, 32, 32, 32)
        path4_1x3 = self.path4_conv1x3(path4)  # output tensor shape: (1, 32, 32, 32)
        path4_3x1 = self.path4_conv3x1(path4)  # output tensor shape: (1, 32, 32, 32)
        path4 = self.concatenate([path4_1x3, path4_3x1])  # output tensor shape: (1, 32, 32, 64)

        # Feature Fusion
        merged = self.concatenate([path1, path2, path3, path4])

        # Classification
        flattened = self.flatten(merged)
        output = self.dense(flattened)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # CIFAR-10 input shape: (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)