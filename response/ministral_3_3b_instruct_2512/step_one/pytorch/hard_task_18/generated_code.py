import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: two 3x3 convolutional layers followed by average pooling
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )

        # Second block: fully connected layers for channel weight refinement
        self.fc_block = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(approximate=True),
            nn.Linear(64, 64)
        )

        # Reshape the output of fc_block to match the input dimensions
        self.reshape = nn.Flatten()

        # Final classification layer
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        # First block: feature extraction
        x = self.conv_block(x)

        # Reshape the output to match the input dimensions for addition
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 64)

        # Second block: channel weight refinement
        weight = self.fc_block(x)
        weight = weight.view(x.size(0), 64, 1, 1)  # Reshape to (batch_size, 64, 1, 1)

        # Add the original feature map with the weighted version
        x = torch.add(x.view(x.size(0), 64, 1, 1), weight)

        # Flatten and pass through final classification layer
        x = self.reshape(x)
        x = self.classifier(x)

        return x

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(4, 3, 32, 32)  # Example with batch_size=4
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)