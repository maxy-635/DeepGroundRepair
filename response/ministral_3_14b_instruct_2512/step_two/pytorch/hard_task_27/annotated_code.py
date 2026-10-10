import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Extract spatial features with 7x7 depthwise separable convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=1, padding=3, groups=3),
            nn.LayerNorm([3, 32, 32]),  # Assuming CIFAR-10 image size is 32x32
            nn.ReLU(inplace=True)
        )

        # Two fully connected layers for channel-wise feature transformation
        self.fc1 = nn.Linear(in_features=3*32*32, out_features=3*32*32)
        self.fc2 = nn.Linear(in_features=3*32*32, out_features=3*32*32)

        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(in_features=3*32*32, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=10)
        )

    def forward(self, x):
        # Extract spatial features
        spatial_features = self.depthwise_conv(x)

        # Flatten for fully connected layers
        batch_size = x.size(0)
        spatial_features_flat = spatial_features.view(batch_size, -1)

        # Channel-wise feature transformation
        transformed_features = self.fc1(spatial_features_flat)
        transformed_features = self.fc2(transformed_features)

        # Combine original input with processed features
        original_flat = x.view(batch_size, -1)
        combined_features = torch.add(original_flat, transformed_features)

        # Classification
        output = self.classifier(combined_features)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)