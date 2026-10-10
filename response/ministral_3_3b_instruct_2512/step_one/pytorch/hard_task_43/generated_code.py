import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three parallel paths with different pooling layers
        self.pooling1x1 = nn.AvgPool2d(kernel_size=1, stride=1)
        self.pooling2x2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pooling4x4 = nn.AvgPool2d(kernel_size=4, stride=4)

        # Flatten layers for concatenation
        self.flatten = nn.Flatten()

        # Fully connected layer between Block 1 and Block 2
        self.fc_block1_to_2 = nn.Linear(28 * 28 * 3, 128)  # Assuming 3 channels after concatenation

        # Reshape layer for Block 2 input
        self.reshape = nn.Unflatten(1, (1, 16, 16))  # Adjust dimensions as needed

        # Block 2: Three branches for feature extraction
        # Branch 1: <1x1 convolution, 3x3 convolution>
        self.branch1_conv1x1 = nn.Conv2d(3, 32, kernel_size=1, stride=1)
        self.branch1_conv3x3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)

        # Branch 2: <1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution>
        self.branch2_conv1x1 = nn.Conv2d(3, 32, kernel_size=1, stride=1)
        self.branch2_conv1x7 = nn.Conv2d(32, 32, kernel_size=1, stride=1)
        self.branch2_conv7x1 = nn.Conv2d(32, 32, kernel_size=7, stride=1)
        self.branch2_conv3x3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)

        # Branch 3: Average pooling
        self.branch3_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(16 * 16 * 32, 128)
        self.fc2 = nn.Linear(128, 10)

        # ReLU activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # Block 1: Three parallel paths
        pool1x1 = self.pooling1x1(x)
        pool2x2 = self.pooling2x2(x)
        pool4x4 = self.pooling4x4(x)

        # Flatten and concatenate outputs
        flattened = self.flatten(torch.cat([pool1x1, pool2x2, pool4x4], dim=1))
        flattened = self.relu(flattened)

        # Fully connected layer between Block 1 and Block 2
        fc_output = self.fc_block1_to_2(flattened)
        fc_output = self.relu(fc_output)

        # Reshape for Block 2
        reshaped = self.reshape(fc_output)

        # Block 2: Three branches
        branch1 = self.relu(self.branch1_conv1x1(reshaped))
        branch1 = self.relu(self.branch1_conv3x3(branch1))

        branch2 = self.relu(self.branch2_conv1x1(reshaped))
        branch2 = self.relu(self.branch2_conv1x7(branch2))
        branch2 = self.relu(self.branch2_conv7x1(branch2))
        branch2 = self.relu(self.branch2_conv3x3(branch2))

        branch3 = self.relu(self.branch3_pool(reshaped))

        # Concatenate outputs from all branches
        concatenated = torch.cat([branch1, branch2, branch3], dim=1)

        # Fully connected layers for classification
        fc_output = self.fc1(concatenated)
        fc_output = self.relu(fc_output)
        output = self.fc2(fc_output)

        return output

if __name__ == "__main__":
    # Dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 28, 28)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)