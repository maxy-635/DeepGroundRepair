import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Channel attention path 1: Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1_avg = nn.Linear(in_features=64, out_features=32)
        self.fc2_avg = nn.Linear(in_features=32, out_features=16)

        # Channel attention path 2: Global Max Pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1_max = nn.Linear(in_features=64, out_features=32)
        self.fc2_max = nn.Linear(in_features=32, out_features=16)

        # Combine channel attention paths
        self.add = lambda x, y: x + y
        self.relu = nn.ReLU()

        # Spatial attention path: Average Pooling
        self.avg_pool_spatial = nn.AdaptiveAvgPool2d(1)

        # Spatial attention path: Max Pooling
        self.max_pool_spatial = nn.AdaptiveMaxPool2d(1)

        # Concatenate spatial features
        self.cat = torch.cat

        # Fully connected layer for final classification
        self.fc3 = nn.Linear(in_features=1024, out_features=10)  # Assuming 10 classes for CIFAR-10


    def forward(self, x):
        # Initial convolutional layer
        x = self.conv1(x)

        # Channel attention path 1: Global Average Pooling
        avg_out = self.avg_pool(x)
        avg_out = avg_out.view(avg_out.size(0), -1)  # Flatten
        avg_out = self.fc1_avg(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2_avg(avg_out)

        # Channel attention path 2: Global Max Pooling
        max_out = self.max_pool(x)
        max_out = max_out.view(max_out.size(0), -1)  # Flatten
        max_out = self.fc1_max(max_out)
        max_out = self.relu(max_out)
        max_out = self.fc2_max(max_out)

        # Combine channel attention paths
        channel_attention = self.relu(self.add(avg_out, max_out))

        # Apply channel attention to original features
        x = x * channel_attention.unsqueeze(1).unsqueeze(1)  # Element-wise multiplication

        # Spatial attention path: Average Pooling
        avg_spatial = self.avg_pool_spatial(x)
        avg_spatial = avg_spatial.view(avg_spatial.size(0), -1)

        # Spatial attention path: Max Pooling
        max_spatial = self.max_pool_spatial(x)
        max_spatial = max_spatial.view(max_spatial.size(0), -1)

        # Concatenate spatial features
        spatial_features = self.cat((avg_spatial, max_spatial), dim=1)

        # Flatten and feed into fully connected layer
        flattened_features = spatial_features

        output = self.fc3(flattened_features)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for CIFAR-10 (32x32x3)
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)