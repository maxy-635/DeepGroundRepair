import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Add, Concatenate, Flatten, Dense

class DL_Model(Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block
        self.conv_block1 = self.build_conv_block()
        self.conv_block2 = self.build_conv_block()
        self.conv_parallel = Conv2D(filters=32, kernel_size=(3, 3), padding='same')

    def build_conv_block(self):
        conv_block = tf.keras.Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
            Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
            Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')
        ])
        return conv_block

    def call(self, inputs):
        # First branch
        branch1 = self.conv_block1(inputs)
        # Second branch
        branch2 = self.conv_block2(inputs)
        # Parallel branch
        parallel_branch = self.conv_parallel(inputs)
        # Combine outputs
        combined = tf.math.add(tf.math.add(branch1, branch2), parallel_branch)
        # Concatenate outputs from two blocks
        concatenated = tf.experimental.numpy.concatenate([combined, combined], axis=-1)
        # Flatten and dense layer
        flattened = Flatten()(concatenated)
        output = Dense(units=10, activation='softmax')(flattened)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # MNIST image shape
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)