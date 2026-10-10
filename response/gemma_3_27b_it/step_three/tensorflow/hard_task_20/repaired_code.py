import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.conv1x1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5 = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        # Branch Path
        self.branch_conv1x1 = tf.keras.layers.Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Fusion
        self.add = tf.keras.layers.Add()

        # Classification
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main Path
        x1 = self.conv1x1(inputs)
        x2 = self.conv3x3(inputs)
        x3 = self.conv5x5(inputs)
        main_output = self.concatenate([x1, x2, x3])

        # Branch Path
        branch_output = self.branch_conv1x1(inputs)

        # Fusion
        fused_features = self.add([main_output, branch_output])

        # Classification
        x = self.flatten(fused_features)
        x = self.dense1(x)
        output = self.dense2(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input tensor with shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)