import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First pathway
        self.pathway1_conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pathway1_bn1 = nn.BatchNorm2d(num_features=32)
        self.pathway1_conv2 = nn.Conv2d(in_channels=33, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pathway1_bn2 = nn.BatchNorm2d(num_features=32)
        self.pathway1_conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pathway1_bn3 = nn.BatchNorm2d(num_features=32)

        # Second pathway
        self.pathway2_conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pathway2_bn1 = nn.BatchNorm2d(num_features=64)
        self.pathway2_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pathway2_bn2 = nn.BatchNorm2d(num_features=64)
        self.pathway2_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pathway2_bn3 = nn.BatchNorm2d(num_features=64)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=96*28*28, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

        # ReLU activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Pathway 1 processing
        x1 = self.pathway1_conv1(x)  # output tensor shape: (1, 32, 28, 28)
        x1 = self.pathway1_bn1(x1)  # output tensor shape: (1, 32, 28, 28)
        x1 = self.relu(x1)  # output tensor shape: (1, 32, 28, 28)

        # First block concatenation
        x1 = torch.cat([x, x1], dim=1)  # output tensor shape: (1, 33, 28, 28)

        # Second block
        x1 = self.pathway1_conv2(x1)
        x1 = self.pathway1_bn2(x1)
        x1 = self.relu(x1)
        x1 = torch.cat([x1[:, :32, :, :], x1], dim=1)

        # Third block
        x1 = self.pathway1_conv3(x1)
        x1 = self.pathway1_bn3(x1)
        x1 = self.relu(x1)
        x1 = torch.cat([x1[:, :32, :, :], x1], dim=1)

        # Pathway 2 processing
        x2 = self.pathway2_conv1(x)
        x2 = self.pathway2_bn1(x2)
        x2 = self.relu(x2)

        # First block concatenation
        x2 = torch.cat([x, x2], dim=1)

        # Second block
        x2 = self.pathway2_conv2(x2)
        x2 = self.pathway2_bn2(x2)
        x2 = self.relu(x2)
        x2 = torch.cat([x2[:, :64, :, :], x2], dim=1)

        # Third block
        x2 = self.pathway2_conv3(x2)
        x2 = self.pathway2_bn3(x2)
        x2 = self.relu(x2)
        x2 = torch.cat([x2[:, :64, :, :], x2], dim=1)

        # Merge pathways
        x = torch.cat([x1, x2], dim=1)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # MNIST images are 28x28 with 1 channel
    input_tensor = torch.randn(1, 1, 28, 28)  # batch_size=1, channels=1, height=28, width=28
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Should be [1, 10] for 10 classes