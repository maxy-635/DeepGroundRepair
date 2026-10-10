import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.initial_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

        # Block 1: Channel Attention
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1_avg = nn.Linear(in_features=3, out_features=3)
        self.fc2_avg = nn.Linear(in_features=3, out_features=3)
        self.fc1_max = nn.Linear(in_features=3, out_features=3)
        self.fc2_max = nn.Linear(in_features=3, out_features=3)

        self.sigmoid = nn.Sigmoid()

        # Block 2: Spatial Attention
        self.avg_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size='mask', stride='mask', padding='mask'),
            nn.Conv2d(in_channels=3, out_channels='mask', kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        # Additional branch for channel alignment
        self.conv1x1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Final classification layer
        self.fc_classifier = nn.Linear(in_features=3, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Initial convolution
        x = self.initial_conv(x)  # output tensor shape: (1, 3, 32, 32)

        # Block 1: Channel Attention
        avg_out = self.global_avg_pool(x)  # output tensor shape: (1, 3, 1, 1)
        avg_out = avg_out.view(avg_out.size(0), -1)  # output tensor shape: (1, 3)
        avg_out = self.fc1_avg(avg_out)  # output tensor shape: (1, 3)
        avg_out = self.fc2_avg(avg_out)  # output tensor shape: (1, 3)
        avg_out = avg_out.view(avg_out.size(0), 3, 1, 1)  # output tensor shape: (1, 3, 1, 1)

        max_out = self.global_max_pool(x)  # output tensor shape: (1, 3, 1, 1)
        max_out = max_out.view(max_out.size(0), -1)  # output tensor shape: (1, 3)
        max_out = self.fc1_max(max_out)  # output tensor shape: (1, 3)
        max_out = self.fc2_max(max_out)  # output tensor shape: (1, 3)
        max_out = max_out.view(max_out.size(0), 3, 1, 1)  # output tensor shape: (1, 3, 1, 1)

        channel_attention = self.sigmoid(avg_out + max_out)  # output tensor shape: (1, 3, 1, 1)
        x = x * channel_attention  # output tensor shape: (1, 3, 32, 32)

        # Block 2: Spatial Attention
        avg_pool_out = self.avg_pool(x)  # output tensor shape: (1, 1, 26, 26)
        max_pool_out = self.max_pool(x)  # output tensor shape of this layer should be (1, 1, 26, 26)
        spatial_attention = torch.cat([avg_pool_out, max_pool_out], dim=1)
        spatial_attention = spatial_attention[:, :1, :, :]

        x = x * spatial_attention

        # Additional branch for channel alignment
        x_add = self.conv1x1(x)
        x = x + x_add
        x = F.relu(x)

        # Global average pooling for classification
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc_classifier(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)