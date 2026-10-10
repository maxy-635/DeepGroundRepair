import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 3x3 convolutions
        branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Branch 2: 1x1 convolutions followed by two 3x3 convolutions
        branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Branch 3: Max pooling
        branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        # Concatenate the outputs of the branches
        self.fusion = nn.Sequential(
            branch1,
            branch2,
            branch3,
            nn.Conv2d(in_channels=32*2 + 1, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Flatten and fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Concatenate the outputs of the branches
        branch1_out = self.fusion[0](x)
        branch2_out = self.fusion[1](x)
        branch3_out = self.fusion[2](x)

        fused_features = torch.cat((branch1_out, branch2_out, branch3_out), dim=1)
        fused_features = self.fusion[3](fused_features)
        fused_features = self.fusion[4](fused_features)
        
        # Flatten and pass through fully connected layers
        flattened_features = self.flatten(fused_features)
        fc1_out = self.fc1(flattened_features)
        fc1_out = self.relu(fc1_out)
        output = self.fc2(fc1_out)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)