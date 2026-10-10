import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.transforms import ReshapeTransform

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three max pooling layers with different scales
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))

        # Flatten and concatenate the outputs of the three pooling layers
        self.flatten_concat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 100),  # Assuming MNIST input size is 28x28
            ReshapeTransform((4, 100))  # Reshape to 4D tensor
        )

        # Block 2: Multiple branches for feature extraction
        self.conv1x1 = nn.Conv2d(1, 32, kernel_size=(1, 1))
        self.conv3x3 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.conv5x5 = nn.Conv2d(1, 32, kernel_size=(5, 5))
        self.max_pooling = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        # Concatenate the outputs from all branches
        self.concat_branch = nn.Sequential(
            nn.Conv2d(4 * 32, 64, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 10)  # Assuming the output dimensions after pooling are 7x7
        )

    def forward(self, x):
        # Block 1 processing
        pool1_out = self.pool1(x)
        pool2_out = self.pool2(x)
        pool3_out = self.pool3(x)

        # Flatten and concatenate
        flattened_concat = torch.cat([
            self.flatten_concat(pool1_out),
            self.flatten_concat(pool2_out),
            self.flatten_concat(pool3_out)
        ], dim=1)

        # Block 2 processing
        conv1x1_out = self.conv1x1(x)
        conv3x3_out = self.conv3x3(x)
        conv5x5_out = self.conv5x5(x)
        max_pooling_out = self.max_pooling(x)

        # Concatenate the outputs from all branches
        branch_out = torch.cat([conv1x1_out, conv3x3_out, conv5x5_out, max_pooling_out], dim=1)
        branch_out = self.concat_branch(branch_out)

        return branch_out

if __name__ == "__main__":
    # Dummy input tensor for MNIST (1 channel, 28x28)
    input_tensor = torch.randn(1, 1, 28, 28)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)