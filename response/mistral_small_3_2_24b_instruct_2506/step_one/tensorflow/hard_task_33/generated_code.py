import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, DepthwiseConv2D, Add, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the three branches
        self.branch1 = self._create_branch()
        self.branch2 = self._create_branch()
        self.branch3 = self._create_branch()
        # Define the final layers
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def _create_branch(self):
        # Create a single branch block
        return tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=(1, 1), padding='same'),
            DepthwiseConv2D(kernel_size=(3, 3), padding='same'),
            Conv2D(filters=64, kernel_size=(1, 1), padding='same'),
            Add()
        ])

    def call(self, inputs):
        # Process inputs through each branch
        branch1_output = self.branch1(inputs)
        branch2_output = self.branch2(inputs)
        branch3_output = self.branch3(inputs)
        # Concatenate the outputs
        concatenated = tf.experimental.numpy.concatenate([branch1_output, branch2_output, branch3_output], axis=-1)
        # Flatten and pass through dense layer
        flattened = self.flatten(concatenated)
        output = self.dense(flattened)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # MNIST input shape
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)