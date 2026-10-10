import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.max_pool_1x1 = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.max_pool_2x2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.max_pool_4x4 = tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
        self.flatten_1x1 = tf.keras.layers.Flatten()
        self.flatten_2x2 = tf.keras.layers.Flatten()
        self.flatten_4x4 = tf.keras.layers.Flatten()
        self.concatenate = tf.keras.layers.Concatenate(axis='mask')
        self.dense_1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        x = inputs  # output tensor shape: (1, 32, 32, 3)
        x_1x1 = self.max_pool_1x1(x)  # output tensor shape: (1, 32, 32, 3)
        x_2x2 = self.max_pool_2x2(x)  # output tensor shape: (1, 16, 16, 3)
        x_4x4 = self.max_pool_4x4(x)  # output tensor shape: (1, 8, 8, 3)

        x_1x1_flat = self.flatten_1x1(x_1x1)  # output tensor shape: (1, 3072)
        x_2x2_flat = self.flatten_2x2(x_2x2)  # output tensor shape: (1, 768)
        x_4x4_flat = self.flatten_4x4(x_4x4)  # output tensor shape: (1, 192)

        concatenated = self.concatenate([x_1x1_flatten, x_2x2_flatten, x_4x4_flatten])

        x = self.dense_1(concatenated)
        x = self.dense_2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)