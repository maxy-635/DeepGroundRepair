import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Concatenate, Flatten, Dense

class DL_Model(Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Compress input channels with 1x1 convolution
        self.conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')

        # Parallel convolutional layers
        self.conv1x1_parallel = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='valid')
        self.conv3x3_parallel = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')

        # Concatenate layer
        self.concatenate = Concatenate(axis=-1)

        # Flatten layer
        self.flatten = Flatten()

        # Fully connected layers
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')  # Assuming 10 classes for classification

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 64)
        # Compress input channels
        x = self.conv1x1(inputs)  # output tensor shape: (1, 32, 32, 32)

        # Parallel convolutions
        x1 = self.conv1x1_parallel(x)  # output tensor shape of this layer should be (1, 30, 30, 32)
        x3 = self.conv3x3_parallel(x)  # output tensor shape: (1, 30, 30, 32)

        # Concatenate results
        x = self.concatenate([x1, x3])

        # Flatten and pass through dense layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 64))  # Batch size of 1
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)