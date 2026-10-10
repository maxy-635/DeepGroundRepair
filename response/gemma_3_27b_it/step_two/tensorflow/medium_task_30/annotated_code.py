import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.avg_pool1 = tf.keras.layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.avg_pool2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.avg_pool3 = tf.keras.layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
        self.flatten1 = tf.keras.layers.Flatten()
        self.flatten2 = tf.keras.layers.Flatten()
        self.flatten3 = tf.keras.layers.Flatten()
        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        pool1 = self.avg_pool1(inputs)
        pool2 = self.avg_pool2(inputs)
        pool3 = self.avg_pool3(inputs)

        flat1 = self.flatten1(pool1)
        flat2 = self.flatten2(pool2)
        flat3 = self.flatten3(pool3)

        concat_features = self.concat([flat1, flat2, flat3])
        
        dense1_output = self.dense1(concat_features)
        output = self.dense2(dense1_output)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)