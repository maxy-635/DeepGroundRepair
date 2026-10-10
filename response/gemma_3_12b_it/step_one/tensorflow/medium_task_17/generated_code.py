import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.dense = tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax)

    def call(self, inputs):
        shape = tf.shape(inputs)
        channels = shape[-1]
        groups = 3
        channels_per_group = channels // groups

        reshaped_tensor = tf.reshape(inputs, [shape[0], shape[1], shape[2], groups, channels_per_group])

        perm = [0, 1, 3, 2, 4]
        transposed_tensor = tf.transpose(reshaped_tensor, perm=perm)

        original_shape = tf.shape(inputs)
        reshaped_back_tensor = tf.reshape(transposed_tensor, [original_shape[0], original_shape[1], original_shape[2], channels])

        output = self.dense(reshaped_back_tensor)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)