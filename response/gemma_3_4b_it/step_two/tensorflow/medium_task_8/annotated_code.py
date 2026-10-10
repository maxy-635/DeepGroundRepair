import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding='same'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding='same'
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding='same'
        )
        self.concatenate = tf.keras.layers.concatenate([
            tf.keras.layers.Lambda(lambda x: x),
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=3,
                strides=1,
                padding='same'
            ),
            tf.keras.layers.Lambda(lambda x: x)
        ], axis=-1)
        self.conv4 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=1,
            strides=1,
            padding='same'
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(
            units=10,
            activation='softmax'
        )

    def call(self, inputs):
        main_path = inputs
        main_path = self.conv1(main_path)
        main_path = self.concatenate([main_path, self.conv2(main_path), main_path])
        main_path = self.conv3(main_path)
        branch_path = self.conv4(inputs)
        merged = tf.keras.layers.add([main_path, branch_path])
        return self.dense(merged)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)