import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D, Concatenate, MaxPool2D, Flatten, Dense, Lambda, Multiply, Add

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: split into three groups with different kernel sizes
        self.conv1x1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')

        # Second block: multiple branches for feature extraction
        # Branch 1: 1x1 convolution
        self.branch1_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Branch 2: <1x1 convolution, 3x3 convolution>
        self.branch2_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch2_conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch 3: <1x1 convolution, 3x3 convolution>
        self.branch3_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch3_conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch 4: <max pooling, 1x1 convolution>
        self.branch4_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.branch4_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Concatenate layer for first block
        self.concat1 = Concatenate(axis=-1)

        # Concatenate layer for second block
        self.concat2 = Concatenate(axis=-1)

        # Final layers
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block processing
        x1 = self.conv1x1(inputs)
        x2 = self.conv3x3(inputs)
        x3 = self.conv5x5(inputs)
        block1_output = self.concat1([x1, x2, x3])

        # Second block processing
        # Branch 1
        branch1 = self.branch1_conv1x1(block1_output)

        # Branch 2
        branch2 = self.branch2_conv3x3(self.branch2_conv1x1(block1_output))

        # Branch 3
        branch3 = self.branch3_conv3x3(self.branch3_conv1x1(block1_output))

        # Branch 4
        branch4 = self.branch4_conv1x1(self.branch4_pool(block1_output))

        # Concatenate all branches
        block2_output = self.concat2([branch1, branch2, branch3, branch4])

        # Final layers
        x = self.flatten(block2_output)
        output = self.dense(x)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 input shape: (32, 32, 3)
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)