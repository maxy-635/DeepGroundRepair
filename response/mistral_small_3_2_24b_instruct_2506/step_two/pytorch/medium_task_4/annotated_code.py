import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path1: Two blocks of convolution followed by average pooling
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size='mask', stride='mask', padding='mask')
        )

        # Path2: Single convolutional layer
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Flatten and fully connected layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=64*8*8 + 64*32*32, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process through both paths
        path1_out = self.path1(x)  # output tensor shape of this layer should be (1, 64, 32, 32)
        path2_out = self.path2(x)  # output tensor shape: (1, 64, 32, 32)

        # Combine features by addition
        combined = path1_out + path2_out

        # Flatten and pass through fully connected layer
        flattened = self.flatten(combined)
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor)