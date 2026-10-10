import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path1: Two blocks of convolution followed by average pooling
        self.path1 = nn.Sequential(
            # First block
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            # Second block
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

        # Path2: Single convolutional layer
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Flatten layers
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()

        # Fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Linear(in_features=81920, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process through both paths
        path1_output = self.path1(x)  # output tensor shape: (1, 256, 8, 8)
        path2_output = self.path2(x)  # output tensor shape: (1, 64, 32, 32)

        # Flatten the outputs
        path1_flat = self.flatten1(path1_output)  # output tensor shape: (1, 16384)
        path2_flat = self.flatten2(path2_output)  # output tensor shape: (1, 65536)

        # Combine the features (addition)
        combined_features = torch.cat([path1_flat, path2_flat], dim=1)  # output tensor shape: (1, 81920)

        # Classification
        output = self.fc(combined_features)
        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Should be [1, 10]