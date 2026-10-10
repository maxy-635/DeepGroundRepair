import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Split input into three groups along the channel dimension
        self.conv1x1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5 = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Split the input tensor into three groups
        group1 = inputs[:, :, :, :tf.shape(inputs)[-1] // 3]
        group2 = inputs[:, :, :, tf.shape(inputs)[-1] // 3:2 * tf.shape(inputs)[-1] // 3]
        group3 = inputs[:, :, :, 2 * tf.shape(inputs)[-1] // 3:]

        # Apply different convolutional kernels to each group
        out1 = self.conv1x1(group1)
        out2 = self.conv3x3(group2)
        out3 = self.conv5x5(group3)

        # Concatenate the outputs
        merged = self.concatenate([out1, out2, out3])

        # Flatten and pass through fully connected layers
        flattened = self.flatten(merged)
        dense1_out = self.dense1(flattened)
        output = self.dense2(dense1_out)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor with the expected shape (batch_size, height, width, channels)
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)