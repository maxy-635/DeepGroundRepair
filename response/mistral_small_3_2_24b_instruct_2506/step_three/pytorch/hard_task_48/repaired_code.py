import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1x1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn1x1 = nn.BatchNorm2d(num_features=64)

        self.conv3x3 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3x3 = nn.BatchNorm2d(num_features=64)

        self.conv5x5 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn5x5 = nn.BatchNorm2d(num_features=64)

        # Block 2
        # Path 1
        self.path1_conv1x1 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Path 2
        self.path2_avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.path2_conv1x1 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Path 3
        self.path3_conv1x1 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.path3_conv1x3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.path3_conv3x1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=1, padding=(1, 0))

        # Path 4
        self.path4_conv1x1 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.path4_conv3x3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.path4_conv1x3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.path4_conv3x1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=1, padding=(1, 0))

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=393216, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1
        x1 = self.conv1x1(x)  # output tensor shape: (1, 64, 32, 32)
        x1 = self.bn1x1(x1)  # output tensor shape: (1, 64, 32, 32)

        x2 = self.conv3x3(x)  # output tensor shape: (1, 64, 32, 32)
        x2 = self.bn3x3(x2)  # output tensor shape: (1, 64, 32, 32)

        x3 = self.conv5x5(x)  # output tensor shape: (1, 64, 32, 32)
        x3 = self.bn5x5(x3)  # output tensor shape: (1, 64, 32, 32)

        x = torch.cat([x1, x2, x3], dim=1)  # output tensor shape: (1, 192, 32, 32)

        # Block 2
        # Path 1
        p1 = self.path1_conv1x1(x)  # output tensor shape: (1, 64, 32, 32)

        # Path 2
        p2 = self.path2_avgpool(x)  # output tensor shape: (1, 192, 32, 32)
        p2 = self.path2_conv1x1(p2)  # output tensor shape: (1, 64, 32, 32)

        # Path 3
        p3 = self.path3_conv1x1(x)  # output tensor shape: (1, 64, 32, 32)
        p3a = self.path3_conv1x3(p3)  # output tensor shape: (1, 64, 32, 32)
        p3b = self.path3_conv3x1(p3)  # output tensor shape: (1, 64, 32, 32)
        p3 = torch.cat([p3a, p3b], dim=1)  # output tensor shape: (1, 128, 32, 32)

        # Path 4
        p4 = self.path4_conv1x1(x)  # output tensor shape: (1, 64, 32, 32)
        p4 = self.path4_conv3x3(p4)  # output tensor shape: (1, 64, 32, 32)
        p4a = self.path4_conv1x3(p4)  # output tensor shape: (1, 64, 32, 32)
        p4b = self.path4_conv3x1(p4)  # output tensor shape: (1, 64, 32, 32)
        p4 = torch.cat([p4a, p4b], dim=1)  # output tensor shape: (1, 128, 32, 32)

        # Concatenate all paths
        x = torch.cat([p1, p2, p3, p4], dim=1)  # output tensor shape: (1, 384, 32, 32)

        # Final layers
        x = self.flatten(x)  # output tensor shape: (1, 393216)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)