import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # 1x1 convolution to increase dimensionality of channels threefold
        self.conv1x1 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1, padding=0)

        # 3x3 depthwise separable convolution for initial feature extraction
        self.depthwise_conv = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1)

        # Channel attention mechanism
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features='mask', out_features=9)
        self.fc2 = nn.Linear(in_features=9, out_features=9)
        self.sigmoid = nn.Sigmoid()

        # 1x1 convolution to reduce dimensionality
        self.reduce_conv = nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Increase dimensionality of channels threefold
        x = self.conv1x1(x)  # output tensor shape: (1, 9, 32, 32)

        # Initial feature extraction
        x = self.depthwise_conv(x)  # output tensor shape: (1, 9, 32, 32)

        # Channel attention mechanism
        channel_attention = self.global_avg_pool(x)  # output tensor shape: (1, 9, 1, 1)
        channel_attention = channel_attention.view(channel_attention.size(0), -1)  # output tensor shape: (1, 9)
        channel_attention = self.fc1(channel_attention)  # output tensor shape: (1, 9)
        channel_attention = self.sigmoid(self.fc2(channel_attention))  # output tensor shape: (1, 9)
        channel_attention = channel_attention.view(channel_attention.size(0), 9, 1, 1)  # output tensor shape: (1, 9, 1, 1)

        # Apply channel attention weights
        x = x * channel_attention  # output tensor shape: (1, 9, 32, 32)

        # Reduce dimensionality
        x = self.reduce_conv(x)  # output tensor shape: (1, 3, 32, 32)

        # Combine with initial input (assuming we want to add it)
        x = x + x  # output tensor shape: (1, 3, 32, 32)

        # Flatten and pass through fully connected layer
        x = x.flatten(start_dim=1)  # output tensor shape: (1, 3072)
        x = self.fc1(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print("Input tensor shape:", input_tensor.shape)
    print("Output tensor shape:", output_tensor.shape)