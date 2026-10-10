import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.branch1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same'),
            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')
        ])
        self.branch2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same'),
            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')
        ])
        self.branch3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same'),
            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')
        ])
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        branch1_output = self.branch1(inputs)
        branch2_output = self.branch2(inputs)
        branch3_output = self.branch3(inputs)

        merged_output = tf.keras.layers.concatenate([branch1_output, branch2_output, branch3_output], axis=-1)
        flattened_output = self.flatten(merged_output)
        output = self.dense(flattened_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)