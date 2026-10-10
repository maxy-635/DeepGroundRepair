import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the block with three convolutional layers
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Parallel branch with a single convolutional layer
        self.parallel_branch = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Two blocks for the two parallel branches
        self.block1 = nn.ModuleList([self.block])
        self.block2 = nn.ModuleList([self.block])

        # Flatten and fully connected layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=128*7*7, out_features=10)  # MNIST images are 28x28, after conv layers they'll be 7x7

    def forward(self, x):
        # First block processing
        block1_out = self.block1[0](x)

        # Split block1 output into three paths
        path1 = block1_out[:, :32, :, :]  # First 32 channels
        path2 = block1_out[:, 32:64, :, :]  # Next 32 channels
        path3 = block1_out[:, 64:, :, :]  # Last 64 channels

        # Parallel branch processing
        parallel_out = self.parallel_branch(x)

        # Combine all paths with addition
        combined = torch.add(path1, torch.add(path2, path3))
        combined = torch.add(combined, parallel_out)

        # Second block processing (same as first)
        block2_out = self.block2[0](combined)

        # Split block2 output into three paths
        path1_b2 = block2_out[:, :32, :, :]
        path2_b2 = block2_out[:, 32:64, :, :]
        path3_b2 = block2_out[:, 64:, :, :]

        # Combine all paths with addition
        combined_b2 = torch.add(path1_b2, torch.add(path2_b2, path3_b2))
        combined_b2 = torch.add(combined_b2, parallel_out)

        # Concatenate outputs from both blocks
        concatenated = torch.cat((combined, combined_b2), dim=1)

        # Flatten and pass through fully connected layer
        flattened = self.flatten(concatenated)
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # MNIST images are 1x28x28
    input_tensor = torch.randn(1, 1, 28, 28)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)