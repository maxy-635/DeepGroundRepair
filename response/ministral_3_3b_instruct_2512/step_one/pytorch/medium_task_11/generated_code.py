import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv_layer = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Channel attention path 1: Global Average Pooling + Fully Connected Layers
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc_avg_path1 = nn.Sequential(
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=16, out_features=32)
        )

        # Channel attention path 2: Global Max Pooling + Fully Connected Layers
        self.channel_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.fc_max_path2 = nn.Sequential(
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=16, out_features=32)
        )

        # Spatial attention path: Average and Max Pooling
        self.spatial_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.spatial_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        # Fully connected layer for final output
        self.fc_output = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        # Initial convolutional layer
        x = self.conv_layer(x)

        # Channel attention path 1
        channel_avg_features = self.channel_avg_pool(x)
        channel_avg_features = torch.flatten(channel_avg_features, 1)
        channel_avg_weights = self.fc_avg_path1(channel_avg_features)
        channel_avg_weights = torch.sigmoid(channel_avg_weights)
        x = x * channel_avg_weights.unsqueeze(2).unsqueeze(3)

        # Channel attention path 2
        channel_max_features = self.channel_max_pool(x)
        channel_max_features = torch.flatten(channel_max_features, 1)
        channel_max_weights = self.fc_max_path2(channel_max_features)
        channel_max_weights = torch.sigmoid(channel_max_weights)
        x = x * channel_max_weights.unsqueeze(2).unsqueeze(3)

        # Spatial attention path
        spatial_avg_features = self.spatial_avg_pool(x)
        spatial_max_features = self.spatial_max_pool(x)

        # Concatenate spatial features along channel dimension
        spatial_features = torch.cat([spatial_avg_features, spatial_max_features], dim=1)

        # Combine spatial and channel features
        x = x * spatial_features

        # Flatten and pass through final fully connected layer
        x = torch.flatten(x, 1)
        x = self.fc_output(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor with batch size 1, 3 channels, 32x32 spatial dimensions
    output_tensor = model(input_tensor)
    print("Input tensor shape:", input_tensor.shape)
    print("Output tensor shape:", output_tensor.shape)