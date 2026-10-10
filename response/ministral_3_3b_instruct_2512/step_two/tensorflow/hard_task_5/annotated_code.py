import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.block1_conv1 = tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.block1_conv2 = tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.block1_conv3 = tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.block1_concat = tf.keras.layers.Concatenate(axis=3)

        self.block2_permute = tf.keras.layers.Permute(dims='mask')
        self.block3_depthwise_conv = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=False)

        self.addition = tf.keras.layers.Add()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1
        x1 = self.block1_conv1(inputs)
        x2 = self.block1_conv2(inputs)
        x3 = self.block1_conv3(inputs)
        x = self.block1_concat([x1, x2, x3])

        # Block 2
        shape = tf.shape(x)
        channels = shape[3]
        channels_per_group = channels // 3
        x = tf.reshape(x, (shape[0], shape[1], shape[2], 3, channels_per_group))
        x = self.block2_permute(x)
        x = tf.reshape(x, (shape[0], shape[1], shape[2], channels))

        # Block 3
        x = self.block3_depthwise_conv(x)

        # Branch
        branch = inputs

        # Addition
        output = self.addition([x, branch])

        # Dense layer
        output = self.dense(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # Example input shape for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)