import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: 1x1 convolution
        self.branch1_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.branch2_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')

        # Branch 3: 1x1 convolution followed by 5x5 convolution
        self.branch3_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.branch3_conv2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')

        # Branch 4: 3x3 max pooling followed by 1x1 convolution
        self.branch4_pool = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.branch4_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')

        # Flatten and fully connected layers
        self.flatten = Flatten()
        self.fc1 = Dense(units=1024, activation='relu')
        self.fc2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Branch 1
        branch1 = self.branch1_conv(inputs)

        # Branch 2
        branch2 = self.branch2_conv1(inputs)
        branch2 = self.branch2_conv2(branch2)

        # Branch 3
        branch3 = self.branch3_conv1(inputs)
        branch3 = self.branch3_conv2(branch3)

        # Branch 4
        branch4 = self.branch4_pool(inputs)
        branch4 = self.branch4_conv(branch4)

        # Concatenate all branches
        concatenated = tf.experimental.numpy.concatenate([branch1, branch2, branch3, branch4], axis=-1)

        # Flatten and pass through fully connected layers
        flattened = self.flatten(concatenated)
        fc1 = self.fc1(flattened)
        output = self.fc2(fc1)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)