import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 convolution for dimensionality reduction
        self.branch1 = Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='branch1_conv'
        )

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2 = Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='branch2_conv1'
        )
        self.branch2_conv2 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='branch2_conv2'
        )

        # Branch 3: 1x1 convolution followed by 5x5 convolution
        self.branch3 = Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='branch3_conv1'
        )
        self.branch3_conv2 = Conv2D(
            filters=64,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='branch3_conv2'
        )

        # Branch 4: 3x3 max pooling followed by 1x1 convolution
        self.branch4 = MaxPool2D(
            pool_size=(3, 3),
            strides=(1, 1),
            padding='same',
            name='branch4_pool'
        )
        self.branch4_conv = Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='branch4_conv'
        )

        # Fully connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(
            units=128,
            activation='relu',
            name='dense1'
        )
        self.dense2 = Dense(
            units=10,
            activation='softmax',
            name='dense2'
        )

    def call(self, inputs):
        # Apply branches
        branch1_output = self.branch1(inputs)
        branch2_output = self.branch2(inputs)
        branch2_output = self.branch2_conv2(branch2_output)
        branch3_output = self.branch3(inputs)
        branch3_output = self.branch3_conv2(branch3_output)
        branch4_output = self.branch4(inputs)
        branch4_output = self.branch4_conv(branch4_output)

        # Concatenate branch outputs
        concatenated_output = Concatenate(axis=-1)([branch1_output, branch2_output, branch3_output, branch4_output])

        # Flatten and pass through fully connected layers
        flattened_output = self.flatten(concatenated_output)
        dense1_output = self.dense1(flattened_output)
        output = self.dense2(dense1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor)