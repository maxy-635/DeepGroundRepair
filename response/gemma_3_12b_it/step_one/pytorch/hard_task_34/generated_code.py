import torch
import torch.nn as nn

class FeatureBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv(x)
        x = self.relu(x)
        x = torch.cat([identity, x], dim=1)
        return x


class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.main_path_block1 = FeatureBlock(1, 32)
        self.main_path_block2 = FeatureBlock(32, 64)
        self.main_path_block3 = FeatureBlock(64, 128)
        self.branch_path = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 10)

    def forward(self, x):
        # Main path
        x = self.main_path_block1(x)
        x = self.main_path_block2(x)
        x = self.main_path_block3(x)

        # Branch path
        branch_output = self.branch_path(x)

        # Feature fusion
        fused_features = torch.add(x, branch_output)

        # Flatten and classify
        x = self.flatten(fused_features)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Dummy input (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)