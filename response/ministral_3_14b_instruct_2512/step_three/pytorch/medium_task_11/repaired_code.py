import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Channel attention module
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1_avg = nn.Linear(in_features=64, out_features=32)
        self.fc2_avg = nn.Linear(in_features=32, out_features=64)

        self.fc1_max = nn.Linear(in_features=64, out_features=32)
        self.fc2_max = nn.Linear(in_features=32, out_features=64)

        # Spatial attention module
        self.avg_pool_spatial = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.max_pool_spatial = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Final layers
        self.fc_final = nn.Linear(in_features=128, out_features=10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = F.relu(x)

        # Channel attention
        avg_out = self.avg_pool(x)
        avg_out = avg_out.view(avg_out.size(0), -1)
        avg_out = self.fc1_avg(avg_out)
        avg_out = F.relu(avg_out)
        avg_out = self.fc2_avg(avg_out)
        avg_out = torch.sigmoid(avg_out)
        avg_out = avg_out.view(avg_out.size(0), -1, 1, 1)

        max_out = self.max_pool(x)
        max_out = max_out.view(max_out.size(0), -1)
        max_out = self.fc1_max(max_out)
        max_out = F.relu(max_out)
        max_out = self.fc2_max(max_out)
        max_out = torch.sigmoid(max_out)
        max_out = max_out.view(max_out.size(0), -1, 1, 1)

        channel_attention = avg_out + max_out
        x = x * channel_attention

        # Spatial attention
        avg_spatial = self.avg_pool_spatial(x)
        max_spatial = self.max_pool_spatial(x)

        # Concatenate along channel dimension
        spatial_attention = torch.cat((avg_spatial, max_spatial), dim=1)

        # Combine with channel features
        x = x * spatial_attention

        # Flatten and final classification
        x = x.view(x.size(0), -1)
        x = self.fc_final(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)