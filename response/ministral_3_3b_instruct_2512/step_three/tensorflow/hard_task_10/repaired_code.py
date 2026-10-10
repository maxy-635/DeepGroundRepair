import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, Dense, Flatten

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1x1 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.conv1x7 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same')
        self.conv7x1 = Conv2D(64, kernel_size=(7, 1), strides=(1, 1), padding='same')
        self.conv_align = Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.add_layer = Add()
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Path 1: 1x1 convolution
        path1 = self.conv1x1(inputs)  # output tensor shape: (1, 32, 32, 64)

        # Path 2: sequence of convolutions: 1x1, 1x7, 7x1
        path2 = self.conv1x7(self.conv7x1(self.conv1x7(inputs)))

        # Concatenate outputs and apply 1x1 convolution to align dimensions
        concatenated = self.conv_align(tf.concat([path1, path2], axis=-1))

        # Branch: direct connection to input
        branch = inputs

        # Merge outputs through addition
        merged = self.add_layer([concatenated, branch])

        # Flatten and apply dense layers
        flattened = Flatten()(merged)
        output = self.dense1(flattened)
        output = self.dense2(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)