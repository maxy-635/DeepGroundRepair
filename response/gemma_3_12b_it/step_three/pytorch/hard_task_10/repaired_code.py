import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Feature extraction path 1: 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # Feature extraction path 2: 1x1, 1x7, 7x1 convolutions
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # Concatenate the outputs of path1 and path2
        # and apply a 1x1 convolution to align the channels
        self.concat_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(inplace=True)
        )

        # Branch connecting directly to the input
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(inplace=True)
        )

        # Merge the outputs of the main path and the branch through addition
        # Fully connected layers for classification
        self.fc1 = nn.Linear(in_features=3072, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Feature extraction paths
        path1_out = self.path1(x)  # output tensor shape: (1, 64, 32, 32)
        path2_out = self.path2(x)  # output tensor shape: (1, 64, 32, 32)

        # Concatenate and align channels
        concat_out = torch.cat((path1_out, path2_out), dim=1)  # output tensor shape: (1, 128, 32, 32)
        aligned_out = self.concat_conv(concat_out)  # output tensor shape: (1, 3, 32, 32)

        # Branch
        branch_out = self.branch(x)  # output tensor shape: (1, 3, 32, 32)

        # Merge paths
        merged_out = torch.add(aligned_out, branch_out)  # output tensor shape: (1, 3, 32, 32)

        # Flatten for fully connected layers
        flattened_out = merged_out.view(merged_out.size(0), -1)  # output tensor shape: (1, 3072)

        # Classification
        fc1_out = self.fc1(flattened_out)
        fc1_out = nn.ReLU(inplace=True)(fc1_out)
        output = self.fc2(fc1_out)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)