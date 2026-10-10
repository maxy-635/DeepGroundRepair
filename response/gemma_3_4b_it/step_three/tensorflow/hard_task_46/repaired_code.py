import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')
        self.conv3 = tf.keras.layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same')
        self.concat1 = tf.keras.layers.Concatenate(axis=-1)
        self.conv4 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        self.conv5 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        self.conv6 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.batch_norm1 = tf.keras.layers.BatchNormalization(axis=None)
        self.dense1 = tf.keras.layers.Dense(10, activation='softmax')
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        x = self.conv1(inputs)  # output tensor shape: (1, 32, 32, 32)
        x = self.batch_norm1(x)  # output tensor shape: (1, 32, 32, 32)
        x = tf.keras.layers.Activation('relu')(x)  # output tensor shape: (1, 32, 32, 32)

        x = self.conv2(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.batch_norm1(x)  # output tensor shape: (1, 32, 32, 32)
        x = tf.keras.layers.Activation('relu')(x)  # output tensor shape: (1, 32, 32, 32)

        x = self.conv3(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.batch_norm1(x)  # output tensor shape: (1, 32, 32, 32)
        x = tf.keras.layers.Activation('relu')(x)  # output tensor shape: (1, 32, 32, 32)

        x = self.concat1([x, x, x])  # output tensor shape: (1, 32, 32, 96)

        x = self.conv4(x)  # output tensor shape: (1, 32, 32, 64)
        x = self.batch_norm1(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = self.conv5(x)
        x = self.batch_norm1(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = self.conv6(x)
        x = self.batch_norm1(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = self.max_pool(x)

        x = self.global_avg_pool(x)
        x = self.dense1(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)