import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Global average pooling and fully connected layers
        self.block1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features=32, out_features=32),  # Assuming input channels=32
            nn.Linear(in_features=32, out_features=32)
        )

        # Block 2: Convolutional layers and max pooling
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        )

        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32 * 8 * 8, out_features=128),  # Assuming input size after pooling is 8x8
            nn.Linear(in_features=128, out_features=10)  # 10 classes for CIFAR-10
        )

    def forward(self, x):
        # the initial shape of x is: (1, 32, 32, 32)
        # Block 1 processing
        weights = self.block1(x)  # output tensor shape: (1, 32)
        weights = weights.view(x.size(0), 32, 1, 1)  # output tensor shape: (1, 32, 1, 1)

        # Element-wise multiplication
        weighted_features = torch.mul(x, weights)  # output tensor shape: (1, 32, 32, 32)

        # Block 2 processing
        deep_features = self.block2(x)  # output tensor shape of this layer should be (1, 32, 32, 32)

        # Add the branch from Block 1 to Block 2 output
        fused_features = deep_features + weighted_features

        # Classification
        output = self.classifier(fused_features)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 32, 32, 32)  # Batch size=1, channels=32, height=32, width=32
    output_tensor = model(input_tensor)
    print(output_tensor)