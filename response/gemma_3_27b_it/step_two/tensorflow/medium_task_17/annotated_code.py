import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(shape='mask')
        self.dense_layer = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Get the shape of the input tensor
        input_shape = tf.shape(inputs)
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]

        # Reshape into three groups
        groups = 3
        channels_per_group = channels // groups
        reshaped_tensor = tf.reshape(inputs, shape=(input_shape[0], height, width, groups, channels_per_group))

        # Swap dimensions for channel shuffling
        permuted_tensor = tf.transpose(reshaped_tensor, perm=[0, 1, 2, 4, 3])

        # Reshape back to original input shape
        reshaped_back_tensor = tf.reshape(permuted_tensor, shape=(input_shape[0], height, width, channels))

        # Pass through a fully connected layer with softmax activation
        output_tensor = self.dense_layer(reshaped_back_tensor)
        return output_tensor


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)