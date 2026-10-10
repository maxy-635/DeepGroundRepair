import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_group1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block1_group2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block1_group3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Transition Convolution
        self.transition_conv = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Block 2
        self.global_max_pool = nn.MaxPool2d(kernel_size=1, stride=1)
        self.fc1 = nn.Linear(in_features='mask', out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.fc_classifier = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1
        out1_group1 = self.block1_group1(x)  # output tensor shape: (1, 32, 32, 32)
        out1_group2 = self.block1_group2(x)  # output tensor shape: (1, 32, 32, 32)
        out1_group3 = self.block1_group3(x)  # output tensor shape: (1, 32, 32, 32)
        out1 = torch.concatenate([out1_group1, out1_group2, out1_group3], dim=1)  # output tensor shape: (1, 96, 32, 32)

        # Transition Convolution
        out2 = self.transition_conv(out1)  # output tensor shape: (1, 64, 32, 32)

        # Block 2
        out3 = self.global_max_pool(out2)  # output tensor shape: (1, 64, 32, 32)
        out3 = torch.flatten(out3, 1)  # output tensor shape: (1, 65536)
        weights = F.relu(self.fc1(out3))
        weights = self.fc2(weights)
        weights = torch.reshape(weights, (-1, 64, 1, 1))
        out4 = torch.mul(out2, weights)

        # Branch
        branch_out = x

        # Add main path and branch
        combined_out = out4 + branch_out

        # Classification
        out5 = torch.flatten(combined_out, 1)
        out6 = self.fc_classifier(out5)

        return out6


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)