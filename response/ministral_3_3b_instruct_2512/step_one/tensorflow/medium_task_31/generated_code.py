import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Add, BatchNormalization

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initialize convolutional layers with different kernel sizes
        self.conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same')
        self.bn = BatchNormalization()
        self.flatten = Flatten()
        self.fc1 = Dense(units=128, activation='relu')
        self.fc2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Split along channel dimension
        x = tf.split(inputs, num_or_size_splits=3, axis=3)

        # Process each group with different kernels
        x1 = self.conv1x1(x[0])
        x2 = self.conv3x3(x[1])
        x3 = self.conv5x5(x[2])

        # Concatenate the outputs
        x = tf.concat([x1, x2, x3], axis=3)

        # Add batch normalization
        x = self.bn(x)

        # Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create dummy input tensor with CIFAR-10 shape (32x32x3)
    input_tensor = tf.random.normal((1, 32, 32, 3))

    # Forward pass
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)