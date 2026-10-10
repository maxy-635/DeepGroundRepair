import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Dense, Flatten
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path layers
        self.conv1x1_reduce = Conv2D(filters=64, kernel_size=(1, 1), padding='same')
        self.conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')
        self.conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')

        # Branch path layer
        self.branch_conv = Conv2D(filters='mask', kernel_size=(3, 3), padding='same')

        # Fully connected layers
        self.flatten = Flatten()
        self.fc1 = Dense(units=128, activation='relu')
        self.fc2 = Dense(units=10, activation='softmax')  # Assuming 10 classes for classification

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 64)
        # Main path
        x = self.conv1x1_reduce(inputs)  # output tensor shape: (1, 32, 32, 64)
        x1 = self.conv1x1(x)  # output tensor shape: (1, 32, 32, 64)
        x2 = self.conv3x3(x)  # output tensor shape: (1, 32, 32, 64)
        x_main = tf.experimental.numpy.concatenate([x1, x2], axis=-1)  # output tensor shape: (1, 32, 32, 128)

        # Branch path
        x_branch = self.branch_conv(inputs)  # output tensor shape of this layer should be (1, 32, 32, 128)

        # Combine main and branch paths
        x_combined = tf.math.add(x_main, x_branch)

        # Classification head
        x = self.flatten(x_combined)
        x = self.fc1(x)
        output = self.fc2(x)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 64))  # Batch size of 1
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)