import tensorflow as tf
from tensorflow.keras.layers import Conv1D, DepthwiseConv2D, Conv2D, Add, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the block for each branch
        self.block = tf.keras.Sequential([
            Conv1D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu'),
            DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same', depth_multiplier=1, activation='relu'),
            Conv1D(filters=16, kernel_size=1, strides=1, padding='same', activation='relu')
        ])

        # Create three identical branches
        self.branch1 = self.block
        self.branch2 = self.block
        self.branch3 = self.block

        # Concatenate the outputs of the three branches
        self.concatenate = tf.keras.layers.Concatenate(axis=3)

        # Flatten and dense layers for classification
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Process each branch
        branch1_output = self.branch1(inputs)
        branch2_output = self.branch2(inputs)
        branch3_output = self.branch3(inputs)

        # Concatenate the outputs
        concatenated_output = self.concatenate([branch1_output, branch2_output, branch3_output])

        # Flatten and pass through dense layer
        output = self.flatten(concatenated_output)
        output = self.dense(output)

        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    # For MNIST, we'll use (batch_size, 28, 28, 1)
    input_tensor = tf.random.normal((1, 28, 28, 1))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)