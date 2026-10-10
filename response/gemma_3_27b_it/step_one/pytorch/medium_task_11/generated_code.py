import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        # Channel attention module
        self.avg_pool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=1, stride=1)

        self.fc1_avg = nn.Linear(in_features=64, out_features=32)
        self.relu_avg = nn.ReLU(inplace=True)
        self.fc2_avg = nn.Linear(in_features=32, out_features=64)

        self.fc1_max = nn.Linear(in_features=64, out_features=32)
        self.relu_max = nn.ReLU(inplace=True)
        self.fc2_max = nn.Linear(in_features=32, out_features=64)

        self.sigmoid = nn.Sigmoid()

        # Spatial feature extraction
        self.avg_pool_spatial = nn.AvgPool2d(kernel_size=2, stride=2)
        self.max_pool_spatial = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer for classification
        self.fc_classifier = nn.Linear(in_features=64 * 4 * 4, out_features=10)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.relu1(x)

        # Channel attention
        avg_out = self.avg_pool(x)
        avg_out = torch.flatten(avg_out, 1)
        avg_out = self.fc1_avg(avg_out)
        avg_out = self.relu_avg(avg_out)
        avg_out = self.fc2_avg(avg_out)

        max_out = self.max_pool(x)
        max_out = torch.flatten(max_out, 1)
        max_out = self.fc1_max(max_out)
        max_out = self.relu_max(max_out)
        max_out = self.fc2_max(max_out)

        channel_attention = self.sigmoid(avg_out + max_out)
        channel_attention = channel_attention.view(-1, 64, 1, 1)
        x = torch.mul(x, channel_attention)

        # Spatial feature extraction
        avg_out_spatial = self.avg_pool_spatial(x)
        max_out_spatial = self.max_pool_spatial(x)
        spatial_features = torch.cat((avg_out_spatial, max_out_spatial), dim=1)

        # Combine channel and spatial features
        x = torch.mul(x, spatial_features)

        # Flatten and classify
        x = torch.flatten(x, 1)
        x = self.fc_classifier(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Dummy input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)