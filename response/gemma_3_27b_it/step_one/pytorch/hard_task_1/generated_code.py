import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

        # Block 1: Channel Attention
        self.avg_pool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=1, stride=1)
        self.fc1_avg = nn.Linear(in_features=3, out_features=3)
        self.fc2_avg = nn.Linear(in_features=3, out_features=3)
        self.fc1_max = nn.Linear(in_features=3, out_features=3)
        self.fc2_max = nn.Linear(in_features=3, out_features=3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Block 2: Spatial Attention
        self.avg_pool_spatial = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.max_pool_spatial = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1x1 = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=1, stride=1, padding=0)

        # Additional Branch
        self.conv1x1_branch = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Final Classification
        self.fc_classifier = nn.Linear(in_features=3, out_features=10)

    def forward(self, x):
        # Initial Convolution
        x = self.conv1(x)

        # Block 1: Channel Attention
        avg_pooled = self.avg_pool(x)
        avg_pooled = avg_pooled.view(avg_pooled.size(0), -1)
        avg_features = self.fc2_avg(self.relu(self.fc1_avg(avg_pooled)))
        avg_features = avg_features.view(avg_features.size(0), 3, 1, 1)

        max_pooled = self.max_pool(x)
        max_pooled = max_pooled.view(max_pooled.size(0), -1)
        max_features = self.fc2_max(self.relu(self.fc1_max(max_pooled)))
        max_features = max_features.view(max_features.size(0), 3, 1, 1)

        channel_attention = self.sigmoid(avg_features + max_features)
        x = x * channel_attention

        # Block 2: Spatial Attention
        avg_pooled_spatial = self.avg_pool_spatial(x)
        max_pooled_spatial = self.max_pool_spatial(x)
        spatial_features = torch.concatenate([avg_pooled_spatial, max_pooled_spatial], dim=1)
        spatial_features = self.conv1x1(spatial_features)
        spatial_attention = self.sigmoid(spatial_features)
        x = x * spatial_attention

        # Additional Branch
        branch_output = self.conv1x1_branch(x)
        x = x + branch_output
        x = self.relu(x)

        # Final Classification
        x = x.view(x.size(0), -1)
        x = self.fc_classifier(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)