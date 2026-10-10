import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv_block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', 
                                  kernel_initializer='he_normal', bias_initializer='zeros'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU(max_value=1.0)
        ])
        self.conv_block2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                                  kernel_initializer='he_normal', bias_initializer='zeros'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU(max_value=1.0)
        ])
        self.conv_block3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same',
                                  kernel_initializer='he_normal', bias_initializer='zeros'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU(max_value=1.0)
        ])
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv_block1(inputs)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        output = self.dense2(x)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)