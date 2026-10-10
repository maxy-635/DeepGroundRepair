import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Pathway
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')
        self.dropout1 = tf.keras.layers.Dropout(rate=0.5)

        # Branch Pathway
        self.branch_conv = tf.keras.layers.Conv2D(filters=32, kernel_size='mask', strides='mask', padding='mask', activation='relu')

        # Fusion and Classification
        self.add_layer = tf.keras.layers.Add()
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=False)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        # Main Pathway
        x = self.conv1(inputs)  # output tensor shape: (1, 28, 28, 32)
        x = self.conv2(x)  # output tensor shape: (1, 28, 28, 32)
        x = self.conv3(x)  # output tensor shape: (1, 28, 28, 32)
        x = self.pool1(x)  # output tensor shape: (1, 14, 14, 32)
        x = self.dropout1(x)  # output tensor shape: (1, 14, 14, 32)

        # Branch Pathway
        branch = self.branch_conv(inputs)  # output tensor shape of this layer should be (1, 14, 14, 32)

        # Fusion
        x = self.add_layer([x, branch])

        # Classification
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        return self.dense1(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Example input tensor for MNIST (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)