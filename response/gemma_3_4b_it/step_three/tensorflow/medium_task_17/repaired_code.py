import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.reshape = tf.reshape
        self.transpose = tf.transpose
        self.expand_dims = tf.expand_dims
        self.dense = tf.keras.layers.Dense

    def call(self, input_tensor):
        # Reshape the input tensor
        height, width = tf.shape(input_tensor)[0], tf.shape(input_tensor)[1]
        groups = 3
        channels_per_group = tf.cast(tf.shape(input_tensor)[-1] / groups, tf.int32)
        input_tensor = self.reshape(input_tensor, (height, width, groups, channels_per_group))

        # Swap dimensions for channel shuffling
        input_tensor = self.transpose(input_tensor, [0, 2, 1, 3])

        # Reshape back to original shape
        input_tensor = self.reshape(input_tensor, (height, width, -1))

        # Fully connected layer with softmax activation
        output_tensor = self.dense(input_tensor, units=10, activation='softmax')

        return output_tensor


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)