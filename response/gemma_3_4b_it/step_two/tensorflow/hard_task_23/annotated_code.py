import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=1,
            strides=1,
            padding='same',
            activation='relu'
        )
        self.branch1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=3,
                strides=1,
                padding='same',
                activation='relu'
            ),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=3,
                strides=1,
                padding='same',
                activation='relu'
            )
        ])
        self.branch2 = tf.keras.Sequential([
            tf.keras.layers.AveragePooling2D(
                pool_size=2,
                strides=2,
                padding='same'
            ),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=3,
                strides=1,
                padding='same',
                activation='relu'
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=3,
                strides=2,
                padding='same',
                activation='relu'
            )
        ])
        self.branch3 = tf.keras.Sequential([
            tf.keras.layers.AveragePooling2D(
                pool_size=2,
                strides=2,
                padding='same'
            ),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=3,
                strides=1,
                padding='same',
                activation='relu'
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=3,
                strides=2,
                padding='same',
                activation='relu'
            )
        ])
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.conv_final = tf.keras.layers.Conv2D(
            filters=10,
            kernel_size=1,
            strides=1,
            padding='same',
            activation='softmax'
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.branch1(x)
        x = self.branch2(x)
        x = self.branch3(x)
        x = self.concat([x, x])
        x = self.conv_final(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)