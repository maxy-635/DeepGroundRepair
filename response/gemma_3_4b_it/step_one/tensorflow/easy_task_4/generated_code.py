import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv_block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid'),
            tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid'),
            tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid')
        ])
        self.conv_block2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid'),
            tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid'),
            tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid')
        ])
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv_block1(inputs)
        x = self.conv_block2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.output_layer(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Batch size 1, 28x28 image, 1 channel (grayscale)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)