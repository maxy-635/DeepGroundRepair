import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three max pooling layers with different scales
        self.max_pool_1x1 = nn.MaxPool2d(kernel_size=1, stride=1)
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool_4x4 = nn.MaxPool2d(kernel_size=4, stride=4)

        # Flatten and concatenate the outputs
        self.flatten = nn.Flatten()
        self.fc_block1 = nn.Linear(in_features=1029, out_features=1024)
        self.reshape = nn.Sequential(
            nn.Linear(1024, 16*16*4),  # Reshape to 4D tensor (16x16x4)
            nn.Unflatten(1, (4, 16, 16))  # Unflatten to 4D tensor
        )

        # Block 2: Multiple branches with different convolutions and pooling
        self.branch_1x1_conv = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=1, stride=1)
        self.branch_3x3_conv = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch_5x5_conv = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.branch_3x3_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # Concatenate all branches
        self.concat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*4*16*16, 10)  # Assuming output is 10 classes for MNIST
        )

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Block 1
        pool_1x1 = self.max_pool_1x1(x)  # output tensor shape: (1, 1, 28, 28)
        pool_2x2 = self.max_pool_2x2(x)  # output tensor shape: (1, 1, 14, 14)
        pool_4x4 = self.max_pool_4x4(x)  # output tensor shape: (1, 1, 7, 7)

        # Flatten and concatenate
        pool_1x1_flat = self.flatten(pool_1x1)  # output tensor shape: (1, 784)
        pool_2x2_flat = self.flatten(pool_2x2)  # output tensor shape: (1, 196)
        pool_4x4_flat = self.flatten(pool_4x4)  # output tensor shape: (1, 49)

        # Concatenate along a new dimension
        concatenated = torch.cat((pool_1x1_flat, pool_2x2_flat, pool_4x4_flat), dim=1)  # output tensor shape: (1, 1029)

        # Fully connected and reshape
        fc_output = self.fc_block1(concatenated)
        reshaped = self.reshape(fc_output)

        # Block 2
        branch_1x1 = self.branch_1x1_conv(reshaped)
        branch_3x3 = self.branch_3x3_conv(reshaped)
        branch_5x5 = self.branch_5x5_conv(reshaped)
        branch_pool = self.branch_3x3_pool(reshaped)

        # Concatenate all branches
        branches = torch.cat((branch_1x1, branch_3x3, branch_5x5, branch_pool), dim=1)
        output = self.concat(branches)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28
    output_tensor = model(input_tensor)
    print(output_tensor.shape)