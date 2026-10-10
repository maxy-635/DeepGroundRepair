import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1_1 = tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1_2 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv1_3 = tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), padding='same', activation='relu')
        self.concat1 = tf.keras.layers.concatenate(inputs=inputs)
        self.conv_transition = tf.keras.layers.Conv2D(16, (1, 1), strides=(1, 1), padding='same', activation='relu')
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
        self.reshape = tf.keras.layers.Reshape((10,))

    def call(self, inputs):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.concat1

        x = self.conv_transition(x)
        x = self.global_avg_pool(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.reshape(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)