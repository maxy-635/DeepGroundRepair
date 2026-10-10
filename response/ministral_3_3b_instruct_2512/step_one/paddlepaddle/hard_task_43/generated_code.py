import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three parallel paths with different pooling scales
        self.pooling_1x1 = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.pooling_2x2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=1)
        self.pooling_4x4 = nn.AvgPool2D(kernel_size=4, stride=4, padding=2)

        # Flatten layers for concatenation
        self.flatten = nn.Flatten()

        # Fully connected layer between Block 1 and Block 2
        self.fc_block1_to_block2 = nn.Linear(28 * 28 * 3, 128)  # Assuming 3 channels after concatenation

        # Reshape layer for Block 2 input
        self.reshape_block2 = paddle.distribution.ReshapeTransform((128, 4))

        # Block 2: Three branches for feature extraction
        # Branch 1: <1x1 convolution, 3x3 convolution>
        self.conv_branch1_1x1 = nn.Conv2D(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv_branch1_3x3 = nn.Conv2D(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 2: <1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution>
        self.conv_branch2_1x1 = nn.Conv2D(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv_branch2_1x7 = nn.Conv2D(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv_branch2_7x1 = nn.Conv2D(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv_branch2_3x3 = nn.Conv2D(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 3: Average pooling
        self.avg_pool_branch3 = nn.AvgPool2D(kernel_size=3, stride=1, padding=1)

        # Fully connected layers for classification
        self.fc_classification1 = nn.Linear(64 * 4 * 4, 128)
        self.fc_classification2 = nn.Linear(128, 10)

    def forward(self, x):
        # Block 1: Three parallel paths with different pooling scales
        pooled_1x1 = self.pooling_1x1(x)
        pooled_2x2 = self.pooling_2x2(x)
        pooled_4x4 = self.pooling_4x4(x)

        # Flatten and concatenate the outputs
        flattened_1x1 = self.flatten(pooled_1x1)
        flattened_2x2 = self.flatten(pooled_2x2)
        flattened_4x4 = self.flatten(pooled_4x4)

        concatenated = paddle.concat([flattened_1x1, flattened_2x2, flattened_4x4], axis=1)

        # Apply fully connected layer between Block 1 and Block 2
        block1_output = self.fc_block1_to_block2(concatenated)

        # Reshape for Block 2 input
        reshaped_block2_input = self.reshape_block2(block1_output)

        # Block 2: Three branches for feature extraction
        # Branch 1
        branch1_1x1 = self.conv_branch1_1x1(reshaped_block2_input)
        branch1_3x3 = self.conv_branch1_3x3(reshaped_block2_input)
        branch1_output = paddle.concat([branch1_1x1, branch1_3x3], axis=1)

        # Branch 2
        branch2_1x1 = self.conv_branch2_1x1(reshaped_block2_input)
        branch2_1x7 = self.conv_branch2_1x7(branch2_1x1)
        branch2_7x1 = self.conv_branch2_7x1(branch2_1x7)
        branch2_3x3 = self.conv_branch2_3x3(branch2_1x1)
        branch2_output = paddle.concat([branch2_1x1, branch2_1x7, branch2_7x1, branch2_3x3], axis=1)

        # Branch 3
        branch3_output = self.avg_pool_branch3(reshaped_block2_input)

        # Concatenate all branch outputs
        block2_output = paddle.concat([branch1_output, branch2_output, branch3_output], axis=1)

        # Flatten for fully connected layers
        flattened_block2 = self.flatten(block2_output)

        # Classification layers
        classification_output = self.fc_classification1(flattened_block2)
        classification_output = F.leaky_relu(classification_output)
        classification_output = self.fc_classification2(classification_output)

        return classification_output

if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = paddle.randn((1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)