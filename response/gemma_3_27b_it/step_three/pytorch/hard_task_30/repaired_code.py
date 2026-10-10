import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First Block: Dual-Path Structure
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1)
        )
        self.branch_path = nn.Identity()  # Direct connection

        # Second Block: Depthwise Separable Convolutions
        self.group1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.group2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.group3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=3 * 3 * 3 * 3, out_features=128)  # Adjust input features based on CIFAR-10 image size and second block output
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First Block
        main_path_output = self.main_path(x)  # output tensor shape: (1, 3, 32, 32)
        branch_path_output = self.branch_path(x)  # output tensor shape: (1, 3, 32, 32)
        combined_output = main_path_output + branch_path_output  # output tensor shape: (1, 3, 32, 32)

        # Second Block
        group1_output = self.group1(combined_output)  # output tensor shape: (1, 3, 32, 32)
        group2_output = self.group2(combined_output)  # output tensor shape: (1, 3, 32, 32)
        group3_output = self.group3(combined_output)  # output tensor shape: (1, 3, 32, 32)

        concatenated_output = torch.cat((group1_output, group2_output, group3_output), dim=1)  # output tensor shape: (1, 9, 32, 32)

        # Flatten the tensor for fully connected layers
        flattened_output = concatenated_output.view(1, 9, 32, 32)

        # Fully Connected Layers
        fc1_output = F.relu(self.fc1(flattened_output))
        fc2_output = self.fc2(fc1_output)
        output = self.log_softmax(fc2_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)