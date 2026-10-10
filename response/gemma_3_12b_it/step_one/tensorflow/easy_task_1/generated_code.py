import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

class DL_Model(tf.keras.Sequential):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(Flatten())
        self.add(Dense(units=128, activation='relu'))
        self.add(Dense(units=10, activation='softmax'))


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Dummy input tensor for MNIST (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)