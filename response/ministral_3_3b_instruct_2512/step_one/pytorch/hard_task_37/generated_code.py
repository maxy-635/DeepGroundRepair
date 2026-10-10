import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the block with three sequential convolutional layers
        self.block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Define the parallel branch with a convolutional layer
        self.parallel_branch = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Define the second block with three sequential convolutional layers
        self.block2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Define the addition operation for combining outputs from the block and parallel branch
        self.addition = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Define the concatenation layer
        self.concatenate = nn.Sequential(
            nn.Conv2d(128 + 128, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Define the flattening layer
        self.flatten = nn.Flatten()

        # Define the fully connected layer
        self.fc = nn.Linear(256 * 7 * 7, 10)  # MNIST image size is 28x28

    def forward(self, x):
        # First block processing
        block_output = self.block(x)

        # Parallel branch processing
        parallel_branch_output = self.parallel_branch(x)

        # Combine outputs from the block and parallel branch
        combined_output = torch.add(block_output, parallel_branch_output)

        # Second block processing
        block2_output = self.block2(x)

        # Concatenate outputs from the two blocks
        concatenated_output = torch.cat((combined_output, block2_output), dim=1)

        # Apply concatenation layer
        concatenated_output = self.concatenate(concatenated_output)

        # Flatten and pass through fully connected layer
        flattened_output = self.flatten(concatenated_output)
        output = self.fc(flattened_output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Dummy input tensor for MNIST (batch_size=1, channels=1, height=28, width=28)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (1, 10)