import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First sequential block
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # Second sequential block
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=7*7, out_features=10)

    def forward(self, x):
        # Apply first block
        out1 = self.block1(x)

        # Apply second block
        out2 = self.block2(out1)

        # Combine with input through addition
        combined = torch.add(x, out2)

        # Flatten the combined output
        combined_flat = combined.view(combined.size(0), -1)

        # Classify using the fully connected layer
        output = self.fc(combined_flat)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])