import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: 3 convolutional layers followed by max pooling
        self.conv_block1 = tf.keras.Sequential([
            Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        ])

        # Second block: 4 convolutional layers followed by max pooling
        self.conv_block2 = tf.keras.Sequential([
            Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        ])

        # Fully connected layers
        self.flatten = Flatten()
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(256, activation='relu')
        self.fc3 = Dense(10, activation='softmax')  # MNIST has 10 classes

    def call(self, inputs):
        x = self.conv_block1(inputs)
        x = self.conv_block2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # Dummy input tensor with shape (batch_size, height, width, channels)
    # MNIST images are 28x28 grayscale (1 channel)
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = DL_Model()(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)