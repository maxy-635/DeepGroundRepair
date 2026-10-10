import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial Convolutional Layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)

        # Channel Attention Path 1: Global Average Pooling + FC
        self.avg_pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1_1 = nn.Linear(32 * 1 * 1, 16)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2_1 = nn.Linear(16, 8)

        # Channel Attention Path 2: Global Max Pooling + FC
        self.max_pool1 = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1_2 = nn.Linear(32 * 1 * 1, 8)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc2_2 = nn.Linear(8, 8)

        # Channel Attention
        self.add = nn.Add()
        self.sigmoid_out = nn.Sigmoid()

        # Spatial Feature Extraction
        self.avg_pool2 = nn.AdaptiveAvgPool2d((8, 8))
        self.max_pool2 = nn.AdaptiveMaxPool2d((8, 8))

        # Final Fully Connected Layer
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # Initial Convolution
        x = self.conv1(x)
        x = self.relu(x)

        # Channel Attention Paths
        avg_out = self.avg_pool1(x)
        avg_out = self.fc1_1(avg_out)
        avg_out = self.relu1(avg_out)
        avg_out = self.fc2_1(avg_out)

        max_out = self.max_pool1(x)
        max_out = self.fc1_2(max_out)
        max_out = self.relu2(max_out)
        max_out = self.fc2_2(max_out)

        channel_attention = self.add(avg_out, max_out)
        channel_attention = self.sigmoid_out(channel_attention)

        x = x * channel_attention

        # Spatial Feature Extraction
        avg_spatial = self.avg_pool2(x)
        max_spatial = self.max_pool2(x)

        x = torch.cat([avg_spatial, max_spatial], dim=1)

        # Flatten and Combine
        x = torch.flatten(x, 1)

        # Final Output
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)