import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First convolutional layer with 32 filters, 3x3 kernel, ReLU activation
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        # Second convolutional layer with 64 filters, 3x3 kernel, ReLU activation
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        # Third convolutional layer with 128 filters, 3x3 kernel, ReLU activation
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Concatenate layers along channel dimension
        self.concatenate = tf.keras.layers.Concatenate(axis=3)

        # Flatten layer
        self.flatten = tf.keras.layers.Flatten()

        # First dense layer with 512 units, ReLU activation
        self.dense1 = tf.keras.layers.Dense(units=512, activation='relu')
        # Second dense layer with 10 units (for CIFAR-10 classes), softmax activation
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First convolutional layer
        x = self.conv1(inputs)
        # Second convolutional layer concatenated with first layer
        x = self.concatenate([x, inputs])
        x = self.conv2(x)
        # Third convolutional layer concatenated with second layer
        x = self.concatenate([x, inputs])
        x = self.conv3(x)
        # Flatten and pass through dense layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create dummy input tensor with shape (batch_size, height, width, channels)
    # Using (1, 32, 32, 3) as CIFAR-10 images are 32x32 RGB images
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))

    # Forward pass to validate the model
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)