import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.path1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.AveragePooling2D(pool_size='mask', strides='mask', padding='mask')
        ])
        self.path2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        ])
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        path1_output = self.path1(inputs)  # output tensor shape of this layer should be (1, 32, 32, 32)
        path2_output = self.path2(inputs)  # output tensor shape: (1, 32, 32, 16)
        merged = tf.keras.layers.concatenate([path1_output, path2_output], axis=-1)
        flattened = self.flatten(merged)
        output = self.dense(flattened)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)