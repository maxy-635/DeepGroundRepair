import tensorflow as tf
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPool2D, Conv2DTranspose, Flatten, Dense, ReLU, Add

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = tf.keras.Sequential([
            # First block
            ReLU(),
            SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),

            # Second block
            ReLU(),
            SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            MaxPool2D(pool_size='mask', strides='mask', padding='mask')
        ])

        # Branch path
        self.branch_path = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')
        ])

        # Final layers
        self.add = Add()
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        # Main path processing
        main_output = self.main_path(inputs)  # output tensor shape of this layer should be (1, 28, 28, 64)

        # Branch path processing (applied to the output of first block)
        branch_input = self.main_path.layers[1](inputs)  # output tensor shape: (1, 28, 28, 32)
        branch_output = self.branch_path(branch_input)  # output tensor shape: (1, 28, 28, 64)

        # Add main and branch outputs
        combined = self.add([main_output, branch_output])

        # Final layers
        flattened = self.flatten(combined)
        output = self.dense(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()

    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    # MNIST images are 28x28 with 1 channel
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))

    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Should be (1, 10)