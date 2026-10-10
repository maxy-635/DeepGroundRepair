import torch
import torch.nn as nn
import torch.nn.parameter as parameter

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the convolutional block
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Parallel convolutional branch
        self.parallel_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Second block (same as the first)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Fully connected layer
        self.fc = nn.Linear(in_features=64 * 7 * 7, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # First block
        x1 = self.block(x)  # output tensor shape of this layer should be (1, 32, 28, 28)

        # Parallel branch
        x2 = self.parallel_conv(x)  # output tensor shape: (1, 32, 28, 28)

        # Combine the outputs
        combined = torch.add(x1, x2)

        # Second block
        x3 = self.block2(combined)

        # Concatenate the outputs from the two blocks
        concatenated = torch.cat((x1, x3), dim=1)

        # Flatten the output
        flattened = nn.Flatten()(concatenated)

        # Fully connected layer
        output = self.fc(flattened)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (1, 1, 28, 28) - batch_size, channels, height, width
    input_tensor = torch.randn(1, 1, 28, 28)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)