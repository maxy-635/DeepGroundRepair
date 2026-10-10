import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = tf.keras.layers.concatenate([inputs, x], axis=-1)
        x = self.conv2(x)
        x = tf.keras.layers.concatenate([x, inputs], axis=-1)
        x = self.conv3(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=256, activation='relu')(x)
        x = tf.keras.layers.Dense(units=10, activation='softmax')(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)