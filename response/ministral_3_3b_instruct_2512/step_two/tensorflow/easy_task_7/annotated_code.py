import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.dropout1 = tf.keras.layers.Dropout(rate=0.2)
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.dropout2 = tf.keras.layers.Dropout(rate=0.2)
        self.conv3 = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')

        # Branch path
        self.branch = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')

        # Final layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main path
        x = self.conv1(inputs)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.conv3(x)

        # Branch path
        branch = self.branch(inputs)

        # Combine paths
        combined = tf.keras.layers.Add()([x, branch])

        # Final layers
        x = self.flatten(combined)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    # MNIST images are 28x28 with 1 channel (grayscale)
    input_tensor = tf.random.normal((1, 28, 28, 1))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)