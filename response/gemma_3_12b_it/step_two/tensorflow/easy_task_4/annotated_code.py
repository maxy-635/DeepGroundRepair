import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

class DL_Model(tf.keras.Sequential):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: Three convolutional layers followed by max pooling
        self.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        # Second block: Four convolutional layers followed by max pooling
        self.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        # Feature extraction to classification
        self.add(Flatten())
        self.add(Dense(units=128, activation='relu'))
        self.add(Dense(units=64, activation='relu'))
        self.add(Dense(units=10, activation='softmax'))  # Assuming 10 classes for MNIST


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor with shape (1, 28, 28, 1) - Batch size 1, 28x28 image, 1 channel (grayscale)
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)