import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 3x3 Convolution
        branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Branch 2: 5x5 Convolution
        branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )

        # Attention Mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Linear(in_features=64, out_features=10),
            nn.Softmax(dim=1)
        )

        # Weighted Output Layer
        self.fc = nn.Linear(in_features=64, out_features=10)

        self.branch1 = branch1
        self.branch2 = branch2

    def forward(self, x):
        # Feature Extraction
        out1 = self.branch1(x)
        out2 = self.branch2(x)

        # Combine Features
        combined_features = out1 + out2

        # Attention Weights
        attention_weights = self.attention(combined_features)

        # Weighted Output
        weighted_out1 = out1 * attention_weights
        weighted_out2 = out2 * attention_weights
        weighted_combined = weighted_out1 + weighted_out2

        # Global Average Pooling
        pooled_features = torch.mean(weighted_combined, dim=(2, 3))

        # Fully Connected Layer
        output = self.fc(pooled_features)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)