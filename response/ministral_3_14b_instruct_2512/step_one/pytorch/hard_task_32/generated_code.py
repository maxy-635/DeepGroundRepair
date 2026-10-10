import torch
import torch.nn as nn
import torch.nn.functional as F

class SpecializedBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpecializedBlock, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.Dropout2d(p=0.2)
        )

    def forward(self, x):
        return self.depthwise_conv(x)

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.branches = nn.ModuleList([
            SpecializedBlock(in_channels=1, out_channels=32),
            SpecializedBlock(in_channels=1, out_channels=64),
            SpecializedBlock(in_channels=1, out_channels=128)
        ])
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=32 + 64 + 128, out_features=256),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=10)
        )

    def forward(self, x):
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        concatenated = torch.cat(branch_outputs, dim=1)
        flattened = concatenated.view(concatenated.size(0), -1)
        return self.fc_layers(flattened)

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)