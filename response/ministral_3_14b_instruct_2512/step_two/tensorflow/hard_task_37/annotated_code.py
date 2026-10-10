import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, Concatenate, Flatten, Dense, Lambda, Activation, BatchNormalization, MaxPool2D
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the convolutional block
        self.conv_block = tf.keras.Sequential([
            Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        ])

        # Define the parallel convolutional layer
        self.parallel_conv = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Define the addition layer
        self.add = Add()

        # Define the concatenation layer
        self.concat = Concatenate(axis=-1)

        # Define the flatten and dense layers
        self.flatten = Flatten()
        self.dense = Dense(10, activation='softmax')

    def call(self, inputs):
        # First block
        block1_input = inputs
        block1_conv = self.conv_block(block1_input)

        # Split the block1_conv into three paths
        path1 = block1_conv
        path2 = block1_conv
        path3 = block1_conv

        # Parallel branch
        parallel_branch = self.parallel_conv(inputs)

        # Add parallel branch to each path
        path1 = self.add([path1, parallel_branch])
        path2 = self.add([path2, parallel_branch])
        path3 = self.add([path3, parallel_branch])

        # Second block (same architecture)
        block2_input = inputs
        block2_conv = self.conv_block(block2_input)

        # Split the block2_conv into three paths
        path4 = block2_conv
        path5 = block2_conv
        path6 = block2_conv

        # Add parallel branch to each path
        path4 = self.add([path4, parallel_branch])
        path5 = self.add([path5, parallel_branch])
        path6 = self.add([path6, parallel_branch])

        # Concatenate all paths
        concatenated = self.concat([path1, path2, path3, path4, path5, path6])

        # Flatten and dense layers
        flattened = self.flatten(concatenated)
        output = self.dense(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((32, 28, 28, 1))  # Batch of 32 MNIST images (28x28 grayscale)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)