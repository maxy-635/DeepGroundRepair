import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Depthwise separable convolution layer
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=1, padding=3, groups=3),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm(normalized_shape=[64, 32, 32], eps=1e-05, elementwise_affine=True)
        )

        # Fully connected layers for channel-wise feature transformation
        self.fc1 = nn.Linear(in_features=65536, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)

        # Final fully connected layers for classification
        self.fc3 = nn.Linear(in_features=64, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Extract spatial features
        spatial_features = self.depthwise_conv(x)  # output tensor shape: (1, 64, 32, 32)

        # Flatten spatial features for fully connected layers
        batch_size = spatial_features.size(0)  # output tensor shape: (1)
        spatial_features_flat = spatial_features.view(batch_size, -1)  # output tensor shape: (1, 65536)

        # Channel-wise feature transformation
        transformed_features = self.fc1(spatial_features_flat)
        transformed_features = self.fc2(transformed_features)

        # Reshape back to original dimensions
        transformed_features = transformed_features.view(batch_size, 64, 32, 32)

        # Combine original input with processed features
        combined_features = torch.add(x, transformed_features)

        # Flatten for final classification
        combined_features_flat = combined_features.view(batch_size, -1)

        # Final classification
        output = self.fc3(combined_features_flat)
        output = self.fc4(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 image shape: (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor)