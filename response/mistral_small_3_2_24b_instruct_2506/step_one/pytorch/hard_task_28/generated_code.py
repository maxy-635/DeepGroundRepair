import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.depthwise_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=1, padding=3)
        self.layer_norm = nn.LayerNorm(normalized_shape=[3, 32, 32], eps=1e-05, elementwise_affine=True)

        self.pointwise_conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.pointwise_conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Branch path (identity connection)
        self.branch = nn.Identity()

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=3*32*32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Main path
        x_main = self.depthwise_conv(x)
        x_main = self.layer_norm(x_main)
        x_main = self.pointwise_conv1(x_main)
        x_main = self.pointwise_conv2(x_main)

        # Branch path
        x_branch = self.branch(x)

        # Combine paths
        x_combined = torch.add(x_main, x_branch)

        # Classification
        x_flat = self.flatten(x_combined)
        x_fc1 = self.fc1(x_flat)
        x_fc2 = self.fc2(x_fc1)

        return x_fc2

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)