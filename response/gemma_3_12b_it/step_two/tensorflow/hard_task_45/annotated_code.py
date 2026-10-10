import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, Concatenate, MaxPool2D, Flatten, Dense, Lambda, Input

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First Block
        self.depthwise_conv1_1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.depthwise_conv1_3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.depthwise_conv1_5 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.concat1 = Concatenate(axis=-1)

        # Second Block - Branches
        self.conv2_branch1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv2_branch1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv2_branch1_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2_branch1_4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        
        self.conv2_branch2_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv2_branch2_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        self.max_pool = MaxPool2D(pool_size='mask', strides='mask', padding='mask')
        self.conv2_branch3_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        self.concat2 = Concatenate(axis=-1)

        # Feature Fusion and Classification
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # First Block
        x = self.depthwise_conv1_1(inputs)  # output tensor shape: (1, 32, 32, 3)
        y = self.depthwise_conv1_3(inputs)  # output tensor shape: (1, 32, 32, 3)
        z = self.depthwise_conv1_5(inputs)  # output tensor shape: (1, 32, 32, 3)
        merged = self.concat1([x, y, z])  # output tensor shape: (1, 32, 32, 9)

        # Second Block
        branch1 = self.conv2_branch1_1(merged)  # output tensor shape: (1, 32, 32, 32)
        branch1 = self.conv2_branch1_2(branch1)  # output tensor shape: (1, 32, 32, 32)
        branch1 = self.conv2_branch1_3(branch1)  # output tensor shape: (1, 32, 32, 32)
        branch1 = self.conv2_branch1_4(branch1)  # output tensor shape: (1, 32, 32, 32)

        branch2 = self.conv2_branch2_1(merged)  # output tensor shape: (1, 32, 32, 32)
        branch2 = self.conv2_branch2_2(branch2)  # output tensor shape: (1, 32, 32, 32)

        branch3 = self.max_pool(merged)  # output tensor shape of this layer should be (1, 32, 32, 9)
        branch3 = self.conv2_branch3_1(branch3)

        merged_branches = self.concat2([branch1, branch2, branch3])

        # Feature Fusion and Classification
        x = self.flatten(merged_branches)
        output = self.dense(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)