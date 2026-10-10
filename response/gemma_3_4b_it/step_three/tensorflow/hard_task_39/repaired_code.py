import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1
        self.max_pool_1 = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.max_pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.max_pool_3 = tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
        self.flatten_1 = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu)
        self.reshape_1 = tf.keras.layers.Reshape(target_shape=(1, 128))

        # Block 2
        self.conv_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation=tf.keras.activations.relu)
        self.conv_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=tf.keras.activations.relu)
        self.conv_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation=tf.keras.activations.relu)
        self.max_pool_4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.flatten_2 = tf.keras.layers.Flatten()
        self.dense_2 = tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax)

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        # Block 1
        x = self.max_pool_1(inputs)  # output tensor shape: (1, 28, 28, 1)
        x = self.max_pool_2(x)  # output tensor shape: (1, 14, 14, 1)
        x = self.max_pool_3(x)  # output tensor shape: (1, 3, 3, 1)
        x = self.flatten_1(x)  # output tensor shape: (1, 9)
        x = self.dense_1(x)  # output tensor shape: (1, 128)
        x = self.reshape_1(x)

        # Block 2
        y = self.conv_1(x)
        y = self.conv_2(y)
        y = self.conv_3(y)
        y = self.max_pool_4(y)
        y = self.flatten_2(y)
        output = self.dense_2(y)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)