import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Block 1: Parallel processing paths
        self.path1_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.path1_fc1 = nn.Linear(in_features=3, out_features=3)
        self.path1_fc2 = nn.Linear(in_features=3, out_features=3)

        self.path2_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.path2_fc1 = nn.Linear(in_features=3, out_features=3)
        self.path2_fc2 = nn.Linear(in_features=3, out_features=3)

        # Activation function for channel attention weights
        self.act1 = nn.ReLU(inplace=True)

        # Block 2: Spatial feature extraction
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        # Additional branch with 1x1 convolutional layer
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Final classification layer
        self.fc = nn.Linear(in_features=3, out_features=10)

    def forward(self, x):
        # Initial convolutional layer
        x = self.conv1(x)

        # Block 1: Parallel processing paths
        path1_avg = self.path1_avg_pool(x)
        path1_avg = self.path1_fc1(path1_avg.view(-1, 3))
        path1_avg = self.path1_fc2(path1_avg)
        path1_avg = path1_avg.view(-1, 3, 1, 1)

        path2_max = self.path2_max_pool(x)
        path2_max = self.path2_fc1(path2_max.view(-1, 3))
        path2_max = self.path2_fc2(path2_max)
        path2_max = path2_max.view(-1, 3, 1, 1)

        # Combine the outputs from both paths
        channel_attention = self.act1(path1_avg + path2_max)

        # Apply channel attention weights to the original features
        x = x * channel_attention

        # Block 2: Spatial feature extraction
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        spatial_features = torch.cat([avg_pool, max_pool], dim=1)
        spatial_features = self.conv2(spatial_features)
        spatial_features = self.sigmoid(spatial_features)

        # Apply spatial features to the original features
        x = x * spatial_features

        # Additional branch with 1x1 convolutional layer
        x_branch = self.conv3(x)

        # Add the branch output to the main path
        x = x + x_branch

        # Final classification
        x = x.view(-1, 3)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)