import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D, MaxPool2D, Concatenate, Flatten, Dense, Reshape

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: Split input into three groups and apply depthwise separable convolutions
        self.conv1_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1_3x3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.conv1_5x5 = SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')

        # Concatenate the outputs of the three branches
        self.concatenate1 = Concatenate(axis=-1)

        # Second block: Multiple branches for feature extraction
        self.branch1 = tf.keras.Sequential([
            Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        ])

        self.branch2 = tf.keras.Sequential([
            Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        ])

        self.branch3 = tf.keras.Sequential([
            MaxPool2D(pool_size='mask', strides='mask', padding='mask'),
            Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        ])

        # Concatenate the outputs of the three branches
        self.concatenate2 = Concatenate(axis=-1)

        # Flatten and fully connected layer
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # First block
        x1 = self.conv1_1x1(inputs)  # output tensor shape: (1, 32, 32, 32)
        x2 = self.conv1_3x3(inputs)  # output tensor shape: (1, 32, 32, 3)
        x3 = self.conv1_5x5(inputs)  # output tensor shape: (1, 32, 32, 32)
        x = self.concatenate1([x1, x2, x3])  # output tensor shape: (1, 32, 32, 67)

        # Second block
        branch1_output = self.branch1(inputs)  # output tensor shape: (1, 32, 32, 64)
        branch2_output = self.branch2(inputs)  # output tensor shape: (1, 32, 32, 64)
        branch3_output = self.branch3(inputs)  # output tensor shape of this layer should be (1, 32, 32, 32)
        x = self.concatenate2([branch1_output, branch2_output, branch3_output])

        # Final processing
        x = self.flatten(x)
        output = self.dense(x)

        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)