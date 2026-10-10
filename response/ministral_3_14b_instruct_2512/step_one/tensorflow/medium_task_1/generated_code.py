import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Add, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First convolutional layer
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        # Second convolutional layer
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        # Max pooling layer
        self.maxpool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        # Add layer for skip connection
        self.add = Add()
        # Flatten layer
        self.flatten = Flatten()
        # First dense layer
        self.dense1 = Dense(units=128, activation='relu')
        # Second dense layer
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First convolutional layer
        x = self.conv1(inputs)
        # Second convolutional layer
        x = self.conv2(x)
        # Max pooling layer
        x = self.maxpool(x)
        # Skip connection: add the output of maxpool to the input
        x = self.add([x, inputs])
        # Flatten the output
        x = self.flatten(x)
        # First dense layer
        x = self.dense1(x)
        # Second dense layer
        x = self.dense2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 image shape: 32x32x3
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)