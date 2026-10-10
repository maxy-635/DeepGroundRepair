import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.block1 = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1)),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4)),
            tf.keras.layers.Flatten(),
        ])
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.reshape1 = tf.keras.layers.Reshape((128, 1, 1, 1))
        self.block2 = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(128, 1, 1, 1)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.Flatten(),
        ])
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.block1(inputs)
        x = self.dense1(x)
        x = self.reshape1(x)
        x = self.block2(x)
        x = self.dense2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)