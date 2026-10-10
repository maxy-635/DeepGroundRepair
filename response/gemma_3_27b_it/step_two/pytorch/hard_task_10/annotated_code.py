import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Feature extraction path 1: 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size='mask', stride='mask', padding='mask')
        )

        # Feature extraction path 2: 1x1, 1x7, 7x1 convolutions
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=0)
        )

        # Concatenation and dimension alignment
        self.concat = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=1, stride=1, padding=0) #Align to input channel
        )

        # Branch connecting directly to the input
        self.branch = nn.Identity()  # No operation, just passes the input through

        # Fully connected layers for classification
        self.fc1 = nn.Linear(in_features=3072, out_features=128)  # 32x32x3 = 3072
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Feature extraction
        path1_output = self.path1(x)  # output tensor shape of this layer should be (1, 64, 26, 26)
        path2_output = self.path2(x)  # output tensor shape: (1, 64, 26, 26)

        # Concatenation
        combined_output = torch.concatenate([path1_output, path2_output], dim=1)

        # Dimension alignment
        aligned_output = self.concat(combined_output)

        # Branch and addition
        added_output = torch.add(x, aligned_output)

        # Flatten for fully connected layers
        flattened_output = added_output.view(-1, 3072)

        # Fully connected layers
        fc1_output = F.relu(self.fc1(flattened_output))
        fc2_output = self.fc2(fc1_output)

        return fc2_output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)