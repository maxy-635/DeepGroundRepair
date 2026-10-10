import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1x1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.dropout = nn.Dropout(p=0.5)

        # Block 2
        self.branch1 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch2_conv1 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch2_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch3_conv1 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch3_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.branch4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_conv = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=64 * 32 * 32, out_features=10)  # Assuming input size is 32x32

    def forward(self, x):
        # Block 1
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.dropout(x)

        # Block 2
        branch1 = self.branch1(x)
        branch2 = self.branch2_conv2(self.branch2_conv1(x))
        branch3 = self.branch3_conv2(self.branch3_conv1(x))
        branch4 = self.branch4_conv(self.branch4_pool(x))
        x = torch.cat([branch1, branch2, branch3, branch4], dim=1)

        # Final layers
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 input shape: (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)