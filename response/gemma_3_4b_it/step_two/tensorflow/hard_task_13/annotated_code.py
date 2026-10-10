import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same')
        self.pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding='same')
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.dense1 = tf.keras.layers.Dense(units=16, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)

        x = self.global_avg_pool(x)
        x = self.dense1(x)
        output = self.dense2(x)
        return output


if __name__ == "__main__":
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape: (batch_size, height, width, channels)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print(output_tensor.shape)