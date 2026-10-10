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
        self.fc = nn.Linear(in_features=200704, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # First block processing
        block_output = self.block(x)  # output tensor shape: (1, 128, 28, 28)

        # Parallel branch processing
        parallel_branch_output = self.parallel_branch(x)  # output tensor shape: (1, 128, 28, 28)

        # Combine outputs from the block and parallel branch
        combined_output = torch.add(block_output, parallel_branch_output)  # output tensor shape: (1, 128, 28, 28)

        # Second block processing
        block2_output = self.block2(x)  # output tensor shape: (1, 128, 28, 28)

        # Concatenate outputs from the two blocks
        concatenated_output = torch.cat((combined_output, block2_output), dim=1)  # output tensor shape: (1, 256, 28, 28)

        # Apply concatenation layer
        concatenated_output = self.concatenate(concatenated_output)  # output tensor shape: (1, 256, 28, 28)

        # Flatten and pass through fully connected layer
        flattened_output = self.flatten(concatenated_output)  # output tensor shape: (1, 200704)
        output = self.fc(flattened_output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Dummy input tensor for MNIST (batch_size=1, channels=1, height=28, width=28)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (1, 10)