import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, Concatenate, Flatten, Dense, Lambda

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the block
        self.block = tf.keras.Sequential([
            Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(filters='mask', kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        ])

        # Define the parallel branch
        self.parallel_branch = tf.keras.Sequential([
            Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        ])

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        # Apply the block to the input
        block_output = self.block(inputs)  # output tensor shape of this layer should be (1, 28, 28, 64)

        # Apply the parallel branch to the input
        parallel_output = self.parallel_branch(inputs)  # output tensor shape: (1, 28, 28, 64)

        # Combine the outputs using addition
        combined_output = Add()([block_output, parallel_output])

        # Apply the block again to the combined output
        block_output_2 = self.block(combined_output)

        # Concatenate the outputs from the two blocks
        concatenated_output = Concatenate()([block_output, block_output_2])

        # Flatten and pass through a dense layer
        flattened_output = Flatten()(concatenated_output)
        output = Dense(10, activation='softmax')(flattened_output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Example input shape for MNIST
    output_tensor = model(input_tensor)
    print(output_tensor.shape)