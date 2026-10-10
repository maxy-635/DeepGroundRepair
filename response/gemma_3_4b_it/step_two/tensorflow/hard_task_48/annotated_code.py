import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.block1 = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size='mask', strides='mask', padding='mask', activation='relu'),
            tf.keras.layers.BatchNormalization(axis=-1),
        ])
        self.block2 = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(axis=-1),
        ])
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        block1_output = self.block1(inputs)  # output tensor shape of this layer should be (1, 8, 8, 32)
        block2_output = self.block2(inputs)  # output tensor shape: (1, 8, 8, 64)
        concatenated_output = tf.keras.layers.concatenate([block1_output, block2_output], axis=-1)
        flattened_output = self.flatten(concatenated_output)
        dense_output = self.dense(flattened_output)
        return dense_output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)