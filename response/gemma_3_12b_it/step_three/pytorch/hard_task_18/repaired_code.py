import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # Second block - Main path
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.relu4 = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(in_features=64*64, out_features=10) # CIFAR-10 has 10 classes

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First block
        x = self.conv1(x)  # output tensor shape: (1, 64, 32, 32)
        x = self.relu1(x)  # output tensor shape: (1, 64, 32, 32)
        x = self.conv2(x)  # output tensor shape: (1, 64, 32, 32)
        x = self.relu2(x)  # output tensor shape: (1, 64, 32, 32)
        x = self.avgpool(x)  # output tensor shape: (1, 64, 16, 16)

        # Second block - Main path
        channel_weights = self.adaptive_pool(x)  # output tensor shape: (1, 64, 1, 1)
        channel_weights = self.fc1(channel_weights.reshape(1,64))  # output tensor shape: (1, 64)
        channel_weights = self.relu3(channel_weights)  # output tensor shape: (1, 64)
        channel_weights = self.fc2(channel_weights)  # output tensor shape: (1, 64)
        channel_weights = self.relu4(channel_weights)  # output tensor shape: (1, 64)
        channel_weights = self.flatten(channel_weights)  # output tensor shape: (1, 64)

        # Apply weights
        x = torch.mul(x, channel_weights.expand(x.shape))

        # Classification
        x = self.flatten(x)
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input: batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print(output_tensor.shape)