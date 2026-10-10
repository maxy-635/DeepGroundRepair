import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: 3 Conv2D layers followed by MaxPool2D
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Second block: 4 Conv2D layers followed by MaxPool2D
        self.conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 2), padding='same', activation='relu')
        self.conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv6 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv7 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Fully connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=64, activation='relu')
        self.dense3 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)

        # Second block
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool2(x)

        # Fully connected layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # MNIST images are 28x28 with 1 channel
    output_tensor = model(input_tensor)
    print(output_tensor.shape)