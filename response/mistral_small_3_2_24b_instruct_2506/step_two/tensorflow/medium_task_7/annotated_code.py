import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, Dense, Flatten
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First path: three sequential convolutional layers
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Second path: direct convolution
        self.conv_direct = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Addition layer
        self.add = Add()

        # Fully connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First path
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        # Second path
        x_direct = self.conv_direct(inputs)

        # Add all paths
        x_add = self.add([x3, x_direct])

        # Fully connected layers
        x_flat = self.flatten(x_add)
        x_dense1 = self.dense1(x_flat)
        x_dense2 = self.dense2(x_dense1)

        return x_dense2

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print(output_tensor)