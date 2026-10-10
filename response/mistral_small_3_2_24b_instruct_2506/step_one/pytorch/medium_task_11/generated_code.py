import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Channel attention path 1 (Global Average Pooling)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=64)

        # Channel attention path 2 (Global Max Pooling)
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.fc4 = nn.Linear(in_features=32, out_features=64)

        # Activation function for channel attention
        self.sigmoid = nn.Sigmoid()

        # Spatial attention layers
        self.avg_pool_spatial = nn.AdaptiveAvgPool2d(output_size=1)
        self.max_pool_spatial = nn.AdaptiveMaxPool2d(output_size=1)

        # Final fully connected layer
        self.fc_final = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)

        # Channel attention
        # Path 1: Global Average Pooling
        avg_out = self.avg_pool(x)
        avg_out = avg_out.view(avg_out.size(0), -1)
        avg_out = F.relu(self.fc1(avg_out))
        avg_out = self.fc2(avg_out)
        avg_out = avg_out.view(avg_out.size(0), avg_out.size(1), 1, 1)

        # Path 2: Global Max Pooling
        max_out = self.max_pool(x)
        max_out = max_out.view(max_out.size(0), -1)
        max_out = F.relu(self.fc3(max_out))
        max_out = self.fc4(max_out)
        max_out = max_out.view(max_out.size(0), max_out.size(1), 1, 1)

        # Combine channel attention paths
        channel_attention = avg_out + max_out
        channel_attention = self.sigmoid(channel_attention)
        x = torch.mul(x, channel_attention)

        # Spatial attention
        avg_spatial = self.avg_pool_spatial(x)
        max_spatial = self.max_pool_spatial(x)
        spatial_attention = torch.cat([avg_spatial, max_spatial], dim=1)

        # Combine channel and spatial features
        x = torch.mul(x, spatial_attention)

        # Flatten and final fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc_final(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])