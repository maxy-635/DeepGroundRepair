import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: feature extraction with two 3x3 conv layers and avg pooling
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

        # Second block: channel attention mechanism
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=64),
            nn.Sigmoid()
        )

        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64*16*16, out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        main_path = self.feature_extractor(x)  # output tensor shape: (1, 64, 16, 16)

        # Channel attention path
        channel_weights = self.channel_attention(main_path)  # output tensor shape: (1, 64)
        channel_weights = channel_weights.view(-1, 64, 1, 1)  # output tensor shape: (1, 64, 1, 1)

        # Apply channel attention
        attended_features = main_path * channel_weights  # output tensor shape: (1, 64, 16, 16)

        # Classification
        output = self.classifier(attended_features)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)