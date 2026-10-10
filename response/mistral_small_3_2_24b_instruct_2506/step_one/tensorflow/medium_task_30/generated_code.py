class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.avg_pool1 = tf.keras.layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.avg_pool2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.avg_pool4 = tf.keras.layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
        self.flatten1 = tf.keras.layers.Flatten()
        self.flatten2 = tf.keras.layers.Flatten()
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Apply three average pooling layers
        pool1 = self.avg_pool1(inputs)
        pool2 = self.avg_pool2(inputs)
        pool4 = self.avg_pool4(inputs)

        # Flatten each pooling output
        flat1 = self.flatten1(pool1)
        flat2 = self.flatten1(pool2)
        flat4 = self.flatten1(pool4)

        # Concatenate the flattened features
        concatenated = self.concatenate([flat1, flat2, flat4])

        # Further flatten and process through dense layers
        flattened = self.flatten2(concatenated)
        dense1 = self.dense1(flattened)
        output = self.dense2(dense1)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor)