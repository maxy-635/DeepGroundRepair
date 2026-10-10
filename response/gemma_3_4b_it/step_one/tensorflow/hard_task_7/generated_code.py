import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')
        self.block1_input_split = tf.keras.layers.Split(num_or_size_splits=2, axis=-1)
        self.block1_group1_conv1 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')
        self.block1_group1_depthwise = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.block1_group1_conv2 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')
        self.block1_group2 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)
        self.block2_reshape = tf.keras.layers.Reshape((64, 64, 32, 32))
        self.block2_permute = tf.keras.layers.Permute([2, 3])
        self.block2_reshape2 = tf.keras.layers.Reshape((64, 64, 32))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10)

    def call(self, inputs):
        x = self.conv1(inputs)
        x1, x2 = self.block1_input_split(x)
        x1 = self.block1_group1_conv1(x1)
        x1 = self.block1_group1_depthwise(x1)
        x1 = self.block1_group1_conv2(x1)
        x2 = self.block1_group2(x2)
        x = self.concatenate([x1, x2])
        x = self.block2_reshape(x)
        x = self.block2_permute(x)
        x = self.block2_reshape2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)