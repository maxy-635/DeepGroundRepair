import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First convolutional layer
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        # First max pooling layer
        self.maxpool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        # Second convolutional layer
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        # Flatten layer
        self.flatten = Flatten()
        # First dense layer
        self.dense1 = Dense(units=128, activation='relu')
        # Second dense layer
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # MNIST images are 28x28 with 1 channel
    output_tensor = model(input_tensor)
    print(output_tensor.shape)