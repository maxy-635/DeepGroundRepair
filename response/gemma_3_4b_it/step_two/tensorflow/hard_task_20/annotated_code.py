import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same')
        self.concat = tf.keras.layers.concatenate(inputs='mask', axis='mask')
        self.conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.dense1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.concat([x])
        x = self.conv4(x)
        x = self.global_avg_pool(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)