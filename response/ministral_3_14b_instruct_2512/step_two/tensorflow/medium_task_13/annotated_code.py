import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Flatten, Dense
from tensorflow.keras import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First convolutional layer
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        # Second convolutional layer
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        # Third convolutional layer
        self.conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        # Concatenate layers
        self.concat = Concatenate(axis='mask')
        # Flatten layer
        self.flatten = Flatten()
        # Fully connected layers
        self.dense1 = Dense(units=256, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # First convolutional layer
        x = self.conv1(inputs)  # output tensor shape: (1, 32, 32, 32)
        # Concatenate input with output of first conv layer
        x = self.concat([inputs, x])  # output tensor shape: (1, 32, 32, 35)
        # Second convolutional layer
        x = self.conv2(x)  # output tensor shape: (1, 32, 32, 64)
        # Concatenate previous concatenated output with output of second conv layer
        x = self.concat([x, self.conv2(x)])
        # Third convolutional layer
        x = self.conv3(x)
        # Concatenate previous concatenated output with output of third conv layer
        x = self.concat([x, self.conv3(x)])
        # Flatten the output
        x = self.flatten(x)
        # First fully connected layer
        x = self.dense1(x)
        # Second fully connected layer
        x = self.dense2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 image shape: 32x32x3
    output_tensor = model(input_tensor)
    print(output_tensor.shape)