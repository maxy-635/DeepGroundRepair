import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.dense1 = tf.keras.layers.Dense(
            units=16,
            activation='relu'
        )
        self.reshape1 = tf.keras.layers.Reshape(target_shape='mask')
        self.avg_pool = tf.keras.layers.AveragePooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same'
        )
        self.dense2 = tf.keras.layers.Dense(
            units=10,
            activation='softmax'
        )

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        x = self.conv1(inputs)  # output tensor shape: (1, 32, 32, 32)
        x = self.bn1(x)  # output tensor shape: (1, 32, 32, 32)
        x = tf.keras.layers.ReLU(max_value=0)(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.global_avg_pool(x)  # output tensor shape: (1, 1, 1, 32)
        x = self.dense1(x)  # output tensor shape: (1, 1, 1, 16)
        x = self.reshape1(x)
        inputs = tf.expand_dims(inputs, axis=1)
        x = tf.keras.layers.Concatenate(axis=1)([x, inputs])
        x = self.avg_pool(x)
        x = self.dense2(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)