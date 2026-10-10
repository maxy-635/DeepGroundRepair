import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First sequential block: Conv2d -> MaxPool2d
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # Second sequential block: Conv2d -> MaxPool2d
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size='mask', stride='mask', padding='mask')
        )

        # Ensure the last convolution output dimension matches the input channel dimension
        # Here, we use a 1x1 convolution to adjust the channel dimension to 1
        self.adjust_channels = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)

        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=7*7, out_features=10)  # MNIST images are 28x28, after two max poolings: 7x7

        # Flatten layer
        self.flatten = nn.Flatten()

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # First block
        x1 = self.block1(x)  # output tensor shape: (1, 32, 14, 14)

        # Second block
        x2 = self.block2(x1)  # output tensor shape of this layer should be (1, 64, 28, 28)

        # Adjust channels to match input
        x2_adjusted = self.adjust_channels(x2)

        # Add the adjusted output to the original input
        combined = torch.add(x, x2_adjusted)

        # Flatten and pass through fully connected layer
        flattened = self.flatten(combined)
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])