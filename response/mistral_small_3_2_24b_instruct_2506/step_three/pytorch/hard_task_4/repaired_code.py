import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial 1x1 convolution to increase channel dimensionality
        self.conv1x1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Depthwise separable convolution for feature extraction
        self.depthwise_conv = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1, groups=9)

        # Global average pooling for channel attention
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # Fully connected layers for channel attention
        self.fc1 = nn.Linear(in_features=9, out_features=9 // 2)
        self.fc2 = nn.Linear(in_features=9 // 2, out_features=9)

        # 1x1 convolution to reduce dimensionality
        self.conv_reduce = nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Flatten and fully connected layer for classification
        self.flatten = nn.Flatten()
        self.fc_classifier = nn.Linear(in_features=3 * 32 * 32, out_features=10)  # Assuming input size is 32x32

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Initial 1x1 convolution
        x_initial = self.conv1x1(x)  # output tensor shape of this layer should be (1, 3, 32, 32)

        # Depthwise separable convolution
        x_features = self.depthwise_conv(x_initial)  # output tensor shape: (1, 9, 32, 32)

        # Channel attention
        # Global average pooling
        x_pool = self.global_avg_pool(x_features)  # output tensor shape: (1, 9, 1, 1)
        x_pool = x_pool.view(x_pool.size(0), -1)  # output tensor shape: (1, 9)

        # Fully connected layers
        x_attention = self.fc1(x_pool)  # output tensor shape: (1, 4)
        x_attention = torch.relu(x_attention)  # output tensor shape: (1, 4)
        x_attention = self.fc2(x_attention)  # output tensor shape: (1, 9)
        x_attention = torch.sigmoid(x_attention)  # output tensor shape: (1, 9)

        # Reshape and multiply for channel attention
        x_attention = x_attention.view(x_attention.size(0), x_attention.size(1), 1, 1)  # output tensor shape: (1, 9, 1, 1)
        x_attention = x_features * x_attention  # output tensor shape: (1, 9, 32, 32)

        # 1x1 convolution to reduce dimensionality
        x_reduced = self.conv_reduce(x_attention)  # output tensor shape: (1, 3, 32, 32)

        # Combine with initial input
        x_combined = x_reduced + x_initial

        # Flatten and classify
        x_flat = self.flatten(x_combined)
        x_output = self.fc_classifier(x_flat)

        return x_output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)