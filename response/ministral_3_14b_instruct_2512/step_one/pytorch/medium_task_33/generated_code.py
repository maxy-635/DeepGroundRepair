import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Channel group 1: 1x1 separable convolution
        self.conv1x1_group = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # Channel group 2: 3x3 separable convolution
        self.conv3x3_group = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # Channel group 3: 5x5 separable convolution
        self.conv5x5_group = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # Concatenate the outputs from the three groups
        self.concat = nn.Sequential()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64*3*32*32, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        # Split the input into three channel groups
        x1 = self.conv1x1_group(x)
        x2 = self.conv3x3_group(x)
        x3 = self.conv5x5_group(x)

        # Concatenate along the channel dimension
        x_concat = torch.cat((x1, x2, x3), dim=1)

        # Flatten the concatenated features
        x_flat = x_concat.view(x_concat.size(0), -1)

        # Pass through fully connected layers
        x_fc1 = self.fc1(x_flat)
        x_fc2 = self.fc2(x_fc1)
        x_fc3 = self.fc3(x_fc2)

        return x_fc3

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)