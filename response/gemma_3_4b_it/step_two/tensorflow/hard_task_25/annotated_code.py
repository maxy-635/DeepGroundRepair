import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size='mask', strides='mask', padding='mask')
        self.branch1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same'),
            tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        ])
        self.branch2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same'),
            tf.keras.layers.UpSampling2D(size=(2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same'),
            tf.keras.layers.UpSampling2D(size=(2, 2)),
            tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        ])
        self.branch3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same'),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same'),
            tf.keras.layers.UpSampling2D(size=(2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same'),
            tf.keras.layers.UpSampling2D(size=(2, 2)),
            tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        ])
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.conv_concat = tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        main_path = self.conv1(inputs)  # output tensor shape of this layer should be (1, 1, 1, 32)
        branch1_output = self.branch1(inputs)  # output tensor shape: (1, 1, 1, 32)
        branch2_output = self.branch2(inputs)  # output tensor shape: (1, 1, 1, 32)
        branch3_output = self.branch3(inputs)  # output tensor shape: (1, 1, 1, 32)

        concat_output = self.concat([main_path, branch1_output, branch2_output, branch3_output])
        concat_output = self.conv_concat(concat_output)

        flattened_output = self.flatten(concat_output)
        output_tensor = self.dense(flattened_output)

        return output_tensor


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)