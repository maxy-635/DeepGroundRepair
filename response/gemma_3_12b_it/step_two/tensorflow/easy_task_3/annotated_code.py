import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First feature extraction block: two (convolution, convolution, max pooling)
        self.conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Second feature extraction block: two (convolution, convolution, convolution, max pooling)
        self.conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Fully connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=64, activation='relu')
        self.dense3 = Dense(units=10, activation='softmax')  # Assuming 10 classes for MNIST

    def call(self, inputs):
        # First block
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.pool1(x)

        # Second block
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.pool2(x)

        # Fully connected layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for MNIST (28x28 grayscale images)
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Batch size of 1
    output_tensor = model(input_tensor)
    print(output_tensor.shape)