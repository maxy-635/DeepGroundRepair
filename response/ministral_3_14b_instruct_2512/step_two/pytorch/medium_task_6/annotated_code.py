import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolution
        self.initial_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Three parallel blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True)
            )
        ])

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(in_features=64*32*32, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=10)
        )

        # Softmax output
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)

        # Store initial output for addition later
        initial_output = x

        # Process through parallel blocks
        block_outputs = []
        for block in self.blocks:
            block_output = block(x)
            block_outputs.append(block_output)

        # Concatenate block outputs
        concatenated = torch.cat(block_outputs, dim=1)

        # Add initial output to concatenated outputs
        x = initial_output + concatenated

        # Flatten and pass through fully connected layers
        x = torch.flatten(x, 1)
        x = self.fc(x)

        # Apply softmax
        x = self.softmax(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)