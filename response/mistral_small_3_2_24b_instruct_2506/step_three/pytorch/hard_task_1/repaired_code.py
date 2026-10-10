import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Block 1: Channel attention
        self.avg_pool1 = nn.AdaptiveAvgPool2d(output_size=1)
        self.max_pool1 = nn.AdaptiveMaxPool2d(output_size=1)

        self.fc1_path1 = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=64)
        )

        self.fc1_path2 = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=64)
        )

        self.relu = nn.ReLU(inplace=True)

        # Block 2: Spatial attention
        self.avg_pool2 = nn.AdaptiveAvgPool2d(output_size=1)
        self.max_pool2 = nn.AdaptiveMaxPool2d(output_size=1)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        # Additional branch
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Final classification layer
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)

        # Block 1: Channel attention
        avg_out = self.avg_pool1(x)
        avg_out = avg_out.view(avg_out.size(0), -1)
        avg_out = self.fc1_path1(avg_out)
        avg_out = avg_out.view(avg_out.size(0), -1, 1, 1)

        max_out = self.max_pool1(x)
        max_out = max_out.view(max_out.size(0), -1)
        max_out = self.fc1_path2(max_out)
        max_out = max_out.view(max_out.size(0), -1, 1, 1)

        channel_attention = avg_out + max_out
        channel_attention = self.relu(channel_attention)
        channel_attention = torch.sigmoid(channel_attention)
        x = x * channel_attention

        # Block 2: Spatial attention
        avg_out = self.avg_pool2(x)
        max_out = self.max_pool2(x)
        spatial_attention = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.conv2(spatial_attention)
        spatial_attention = self.sigmoid(spatial_attention)

        x = x * spatial_attention

        # Additional branch
        x = x + self.conv3(x)

        # Final classification
        x = torch.mean(x, dim=[2, 3])
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 image shape: (3, 32, 32)
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)