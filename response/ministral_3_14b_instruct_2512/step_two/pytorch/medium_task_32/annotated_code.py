import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Split the input into three groups along the last dimension
        # Each group will have 1/3 of the input channels
        self.split_channels = 3

        # Define the three branches with different kernel sizes
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=34, out_channels=34, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(34),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=34, out_channels=34, kernel_size=1, stride=1, padding=0, groups=34),
            nn.BatchNorm2d(34),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=34, out_channels=34, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(34),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=34, out_channels=34, kernel_size=3, stride=1, padding=1, groups=34),
            nn.BatchNorm2d(34),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=34, out_channels=34, kernel_size=3, stride=1, padding=1, groups=34),
            nn.BatchNorm2d(34),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=34, out_channels=34, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(34),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=34, out_channels=34, kernel_size=5, stride=1, padding=2, groups=34),
            nn.BatchNorm2d(34),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=34, out_channels=34, kernel_size=5, stride=1, padding=2, groups=34),
            nn.BatchNorm2d(num_features='mask'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=34, out_channels=34, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(34),
            nn.ReLU(inplace=True)
        )

        # Concatenate the outputs of the three branches
        self.concat = nn.Sequential(
            nn.Conv2d(in_channels=102, out_channels=102, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(102),
            nn.ReLU(inplace=True)
        )

        # Fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Linear(in_features=102 * 8 * 8, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 102, 32, 32)
        # Split the input into three groups along the last dimension
        x1 = x[:, :34, :, :]  # output tensor shape: (1, 34, 32, 32)
        x2 = x[:, 34:68, :, :]  # output tensor shape: (1, 34, 32, 32)
        x3 = x[:, 68:, :, :]  # output tensor shape: (1, 34, 32, 32)

        # Process each branch
        x1 = self.branch1(x1)  # output tensor shape: (1, 34, 32, 32)
        x2 = self.branch2(x2)  # output tensor shape: (1, 34, 32, 32)
        x3 = self.branch3(x3)

        # Concatenate the outputs
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.concat(x)

        # Flatten and pass through the fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 102, 32, 32)  # Batch size 1, 102 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)