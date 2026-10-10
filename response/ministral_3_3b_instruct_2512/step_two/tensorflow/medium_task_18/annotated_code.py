import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Concatenate, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Feature extraction at multiple scales
        self.conv1_1x1 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1_3x3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv1_5x5 = Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')

        self.conv2_1x1 = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv2_3x3 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2_5x5 = Conv2D(128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.pool2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')

        # Concatenate features from different scales
        self.concatenate = Concatenate(axis=-1)

        # Classification layers
        self.flatten = Flatten()
        self.fc1 = Dense(256, activation='relu')
        self.fc2 = Dense(10, activation='softmax')

    def call(self, inputs):
        # First scale feature extraction
        x1_1x1 = self.conv1_1x1(inputs)
        x1_3x3 = self.conv1_3x3(inputs)
        x1_5x5 = self.conv1_5x5(inputs)
        x1 = self.concatenate([x1_1x1, x1_3x3, x1_5x5])
        x1 = self.pool1(x1)

        # Second scale feature extraction
        x2_1x1 = self.conv2_1x1(x1)
        x2_3x3 = self.conv2_3x3(x1)
        x2_5x5 = self.conv2_5x5(x1)
        x2 = self.concatenate([x2_1x1, x2_3x3, x2_5x5])

        # Classification
        x = self.flatten(x2)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create dummy input tensor with CIFAR-10 shape (32, 32, 3)
    input_tensor = tf.random.normal((1, 32, 32, 3))

    # Forward pass
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)