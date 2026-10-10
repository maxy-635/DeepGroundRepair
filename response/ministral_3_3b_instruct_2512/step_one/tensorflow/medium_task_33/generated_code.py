import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1x1 = tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3 = tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5 = tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=10, activation='relu')

    def call(self, inputs):
        x = inputs
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)
        x = tf.concat([x1, x2, x3], axis=-1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)