import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')
        self.global_avg_pool1 = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.dense1_1 = tf.keras.layers.Dense(16, activation='relu')
        self.dense1_2 = tf.keras.layers.Dense(16, activation='relu')
        self.global_max_pool1 = tf.keras.layers.GlobalMaxPool2D()
        self.dense2_1 = tf.keras.layers.Dense(8, activation='relu')
        self.dense2_2 = tf.keras.layers.Dense(8, activation='relu')
        self.avg_pool1 = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2), padding='same')
        self.max_pool1 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same')
        self.concat1 = tf.keras.layers.Concatenate(axis=-1)
        self.conv1d1 = tf.keras.layers.Conv1D(16, 1, strides=1, padding='same')
        self.sigmoid1 = tf.keras.layers.Activation('sigmoid')
        self.dense3 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x1 = self.global_avg_pool1(x)
        x2 = self.global_max_pool1(x)
        x1 = self.dense1_1(x1)
        x1 = self.dense1_2(x1)
        x2 = self.dense2_1(x2)
        x2 = self.dense2_2(x2)
        x_avg = self.avg_pool1(x)
        x_max = self.max_pool1(x)
        x_concat = self.concat1([x_avg, x_max])
        x_concat = self.conv1d1(x_concat)
        x_concat = self.sigmoid1(x_concat)
        x_concat = tf.multiply(x_concat, x1)
        x_final = self.dense3(x_concat)
        return x_final

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)