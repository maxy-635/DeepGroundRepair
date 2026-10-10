import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Main and Branch Paths
        self.main_path_conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.main_path_conv2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.branch_path = nn.Identity()  # Direct connection

        # Block 2: Max Pooling Layers
        self.pool1 = nn.MaxPool2d(kernel_size=1, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.flatten = nn.Flatten()

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=16*1*1 + 8*2*2 + 4*4*4, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Block 1
        main_path_output = F.relu(self.main_path_conv1(x))
        main_path_output = self.main_path_conv2(main_path_output)
        branch_path_output = self.branch_path(x)
        block1_output = main_path_output + branch_path_output

        # Block 2
        pool1_output = self.pool1(block1_output)
        pool2_output = self.pool2(block1_output)
        pool3_output = self.pool3(block1_output)

        flattened_pool1 = self.flatten(pool1_output)
        flattened_pool2 = self.flatten(pool2_output)
        flattened_pool3 = self.flatten(pool3_output)

        block2_output = torch.cat((flattened_pool1, flattened_pool2, flattened_pool3), dim=1)

        # Fully Connected Layers
        fc1_output = F.relu(self.fc1(block2_output))
        output = self.fc2(fc1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])