import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 1))
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.fc3 = nn.Linear(in_features=1600, out_features=10)

    def forward(self, x):
        # Initial feature extraction
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        # Global average pooling and fully connected layers
        x2 = self.pool1(x1)
        x2 = torch.flatten(x2, 1)
        x2 = self.fc1(x2)
        x2 = self.fc2(x2)

        # Reshape and multiply with initial features
        x2 = torch.reshape(x2, (-1, 64, 1, 1))
        x2 = torch.mul(x1, x2)

        # Concatenate with input layer
        x = torch.cat((x1, x2), dim=1)

        # Reduce dimensionality and downsample
        x = self.conv2(x)
        x = self.pool2(x)

        # Final classification
        x = torch.flatten(x, 1)
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 image shape: (3, 32, 32)
    output_tensor = model(input_tensor)
    print(model)