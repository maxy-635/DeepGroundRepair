import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial 1x1 convolution to increase channels threefold
        self.initial_conv = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1, padding=0)

        # Depthwise separable convolution (3x3 depthwise + 1x1 pointwise)
        self.depthwise_conv = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1, groups=9)
        self.pointwise_conv = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=1, stride=1, padding=0)

        # Channel attention mechanism
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features=9, out_features=9)
        self.fc2 = nn.Linear(in_features=9, out_features=9)
        self.sigmoid = nn.Sigmoid()

        # Final 1x1 convolution to reduce dimensionality
        self.final_conv = nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Classification layers
        self.flatten = nn.Flatten()
        self.fc_classifier = nn.Linear(in_features=3*32*32, out_features=10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        # Initial feature extraction
        x_initial = self.initial_conv(x)

        # Depthwise separable convolution
        x_depthwise = self.depthwise_conv(x_initial)
        x_pointwise = self.pointwise_conv(x_depthwise)

        # Channel attention
        channel_attention = self.global_avg_pool(x_pointwise)
        channel_attention = channel_attention.view(channel_attention.size(0), -1)
        channel_attention = self.fc1(channel_attention)
        channel_attention = self.fc2(channel_attention)
        channel_attention = self.sigmoid(channel_attention)
        channel_attention = channel_attention.view(channel_attention.size(0), -1, 1, 1)

        # Apply channel attention
        x_attention = x_pointwise * channel_attention

        # Residual connection
        x_residual = x_attention + x_initial

        # Final 1x1 convolution
        x_final = self.final_conv(x_residual)

        # Classification
        x_flat = self.flatten(x_final)
        output = self.fc_classifier(x_flat)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)