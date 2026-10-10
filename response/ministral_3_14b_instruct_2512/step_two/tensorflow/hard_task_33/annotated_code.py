import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Add, Concatenate, Flatten, Dense, BatchNormalization, Activation
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()
        self.num_classes = num_classes

        # Create three branches with the same block structure
        self.branch1 = self._create_branch_block()
        self.branch2 = self._create_branch_block()
        self.branch3 = self._create_branch_block()

        # Concatenate and classification layers
        self.concatenate = Concatenate(axis=-1)
        self.flatten = Flatten()
        self.dense = Dense(num_classes, activation='softmax')

    def _create_branch_block(self):
        """Create a single branch block with the specified architecture"""
        return tf.keras.Sequential([
            # 1x1 convolution to elevate dimension
            Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None),

            # Batch normalization
            BatchNormalization(axis=-1),

            # Activation
            Activation('relu'),

            # 3x3 depthwise separable convolution
            DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation=None),

            # Batch normalization
            BatchNormalization(axis=-1),

            # Activation
            Activation('relu'),

            # 1x1 convolution to reduce dimension
            Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None),

            # Batch normalization
            BatchNormalization(axis=-1),

            # Add the input to the output (residual connection)
            Add()
        ])

    def call(self, inputs):
        # the initial shape of inputs is: (32, 28, 28, 1)
        # Get outputs from each branch
        branch1_out = self.branch1(inputs)
        branch2_out = self.branch2(inputs)
        branch3_out = self.branch3(inputs)

        # Concatenate the outputs
        concatenated = self.concatenate([branch1_out, branch2_out, branch3_out])

        # Flatten and classify
        flattened = self.flatten(concatenated)
        output = self.dense(flattened)

        return output

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create dummy input tensor with shape (batch_size, height, width, channels)
    # MNIST images are 28x28 with 1 channel
    input_tensor = tf.random.normal(shape=(32, 28, 28, 1))

    # Get model output
    output_tensor = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)