import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the block with three sequential convolutional layers
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        )

        # Define the parallel branch
        self.parallel_branch = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Define the second block (same as the first)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        )

        # Define the parallel branch for the second block
        self.parallel_branch2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Define the flatten and fully connected layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=50176, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # First block
        block_output1 = self.block(x)  # output tensor shape: (1, 32, 28, 28)
        parallel_output1 = self.parallel_branch(x)  # output tensor shape: (1, 32, 28, 28)
        combined_output1 = block_output1 + parallel_output1  # output tensor shape: (1, 32, 28, 28)

        # Second block
        block_output2 = self.block2(combined_output1)  # output tensor shape: (1, 32, 28, 28)
        parallel_output2 = self.parallel_branch2(combined_output1)  # output tensor shape: (1, 32, 28, 28)
        combined_output2 = block_output2 + parallel_output2  # output tensor shape: (1, 32, 28, 28)

        # Concatenate the outputs from both blocks
        concatenated_output = torch.cat((combined_output1, combined_output2), dim=1)  # output tensor shape: (1, 64, 28, 28)

        # Flatten and pass through the fully connected layer
        flattened_output = self.flatten(concatenated_output)  # output tensor shape: (1, 50176)
        output = self.fc(flattened_output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor)