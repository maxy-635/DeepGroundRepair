import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=1, padding='same', activation='relu'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=1, padding='same', activation='relu'
        )
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        # Branch path
        self.conv_branch = tf.keras.layers.Conv2D(
            filters=32, kernel_size=1, strides=1, padding='same', activation='relu'
        )

        # Fusion
        self.add = tf.keras.layers.Add()

        # Classification
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Split the input along the last dimension
        group1 = inputs[:, :, :, :inputs.shape[-1] // 3]
        group2 = inputs[:, :, :, inputs.shape[-1] // 3:2 * inputs.shape[-1] // 3]
        group3 = inputs[:, :, :, 2 * inputs.shape[-1] // 3:]

        # Main path
        x2 = self.conv1(group2)
        x = self.concatenate([group1, x2, group3])
        x = self.conv2(x)
        

        # Branch path
        x_branch = self.conv_branch(inputs)

        # Fusion
        x = self.add([x, x_branch])

        # Classification
        x = self.flatten(x)
        x = self.dense(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # Example input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)