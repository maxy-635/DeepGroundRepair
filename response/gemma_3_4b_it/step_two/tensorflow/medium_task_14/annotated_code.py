import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv_block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', 
                                   kernel_initializer='he_normal', bias_initializer='zeros'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU()
        ])
        self.conv_block2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', 
                                   kernel_initializer='he_normal', bias_initializer='zeros'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU()
        ])
        self.conv_block3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', 
                                   kernel_initializer='he_normal', bias_initializer='zeros'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU()
        ])
        self.conv_branch = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters='mask', kernel_size='mask', strides='mask', padding='mask', kernel_initializer='he_normal', bias_initializer='zeros'),

            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU()
        ])
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        path1 = self.conv_block1(inputs)  # output tensor shape: (1, 32, 32, 32)
        path2 = self.conv_block2(path1)  # output tensor shape: (1, 32, 32, 64)
        path3 = self.conv_block3(path2)  # output tensor shape: (1, 32, 32, 128)
        branch = self.conv_branch(inputs)  # output tensor shape of this layer should be (1, 32, 32, 128)
        merged = tf.keras.layers.Add()([path3, branch])
        output = self.fc1(merged)
        output = self.fc2(output)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)