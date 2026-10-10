import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First level
        self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu1 = tf.keras.layers.ReLU()

        # Second level
        self.conv2_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        self.bn2_1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu2_1 = tf.keras.layers.ReLU()

        self.conv2_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        self.bn2_2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu2_2 = tf.keras.layers.ReLU()


        # Third level
        self.conv3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        self.bn3 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu3 = tf.keras.layers.ReLU()

        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(8, 8), strides=(8, 8), padding='valid')
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First level
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        # Second level
        x1 = self.conv2_1(x)
        x1 = self.bn2_1(x1)
        x1 = self.relu2_1(x1)

        x2 = self.conv2_2(x)
        x2 = self.bn2_2(x2)
        x2 = self.relu2_2(x2)

        x = tf.keras.layers.Add()([x1, x2])

        # Third level
        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.relu3(x3)

        x = tf.keras.layers.Add()([x, x3])

        x = self.avg_pool(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.dense(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)