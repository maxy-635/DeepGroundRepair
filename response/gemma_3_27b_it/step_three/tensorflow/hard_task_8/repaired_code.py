import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1x1_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.depthwise_sep_1 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1, activation='relu')
        self.conv1x1_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')

        self.depthwise_sep_branch = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1, activation='relu')
        self.conv1x1_branch = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')

        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        # Block 2
        self.reshape = tf.keras.layers.Reshape(target_shape=(-1, 4, 32))  # Assuming input shape leads to a height and width after Block 1
        self.permute = tf.keras.layers.Permute(dims=(2, 3, 1))
        self.reshape_back = tf.keras.layers.Reshape(target_shape=(-1, 32))

        # Classification
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1
        x = self.conv1x1_1(inputs)
        x = self.depthwise_sep_1(x)
        x = self.conv1x1_2(x)

        branch = self.depthwise_sep_branch(inputs)
        branch = self.conv1x1_branch(branch)

        x = self.concatenate([x, branch])

        # Block 2
        shape = tf.shape(x)
        height = shape[1]
        width = shape[2]
        x = self.reshape(x)
        x = self.permute(x)
        x = self.reshape_back(x)
        
        # Classification
        x = self.dense(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # Dummy input tensor for MNIST
    output_tensor = model(input_tensor)
    print(output_tensor.shape)