import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu1 = nn.ReLU(inplace=True)

        # Global average pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=3)  # Match channel size of initial features

        # 1x1 convolution and average pooling
        self.conv_1x1 = nn.Conv2d(in_channels=64 + 3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # Final fully connected layer
        self.fc_final = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        # Convolutional layer, batch normalization, and ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Global average pooling
        x = self.adaptive_pool(x)

        # Fully connected layers
        x = self.fc1(x.view(x.size(0), -1))  # Flatten
        x = self.relu1(x)
        x = self.fc2(x)

        # Reshape and multiply
        x = x.view(x.size(0), 3, x.size(2), x.size(3))
        weighted_features = torch.mul(x, self.conv1.weight.data)

        # Concatenate with input
        x = torch.cat((self.conv1(x).data, weighted_features), dim=1)  # Concatenate along channel dimension

        # 1x1 convolution and average pooling
        x = self.conv_1x1(x)
        x = self.avg_pool(x)

        # Final fully connected layer
        x = self.fc_final(x.view(x.size(0), -1))

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)