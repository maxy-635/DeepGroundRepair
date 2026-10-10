import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Block 1: Channel Attention
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1_avg = nn.Linear(in_features=3*32*32, out_features=3)
        self.fc2_avg = nn.Linear(in_features=3, out_features=3)
        self.fc1_max = nn.Linear(in_features=3*32*32, out_features=3)
        self.fc2_max = nn.Linear(in_features=3, out_features=3)

        # Block 2: Spatial Attention
        self.avg_pool2d = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv_spatial = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        # Additional Branch
        self.conv_additional = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Classification
        self.fc_classifier = nn.Linear(in_features=3*32*32, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Initial Convolution
        x = self.conv1(x)  # output tensor shape: (1, 3, 32, 32)
        x = self.relu(x)  # output tensor shape: (1, 3, 32, 32)

        # Block 1: Channel Attention
        avg_out = self.avg_pool(x)  # output tensor shape: (1, 3, 16, 16)
        avg_out = torch.reshape(avg_out, (-1, 3, 16, 16))
        avg_out = self.fc1_avg(avg_out)
        avg_out = self.fc2_avg(avg_out)
        avg_out = torch.reshape(avg_out, (-1, 3, 32, 32))

        max_out = self.max_pool(x)
        max_out = torch.reshape(max_out, (-1, 3 * 32 * 32))
        max_out = self.fc1_max(max_out)
        max_out = self.fc2_max(max_out)
        max_out = torch.reshape(max_out, (-1, 3, 32, 32))

        channel_attention = avg_out + max_out
        channel_attention = torch.sigmoid(channel_attention)
        x = x * channel_attention

        # Block 2: Spatial Attention
        avg_spatial = self.avg_pool2d(x)
        max_spatial = self.max_pool2d(x)
        spatial_features = torch.cat((avg_spatial, max_spatial), dim=1)
        spatial_features = self.conv_spatial(spatial_features)
        spatial_attention = self.sigmoid(spatial_features)
        x = x * spatial_attention

        # Additional Branch
        additional_branch = self.conv_additional(x)
        x = x + additional_branch

        # Flatten and Classify
        x = torch.reshape(x, (-1, 3 * 32 * 32))
        x = self.fc_classifier(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)