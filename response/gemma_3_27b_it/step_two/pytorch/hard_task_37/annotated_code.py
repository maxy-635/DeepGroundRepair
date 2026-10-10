import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the convolutional block
        block = nn.Sequential(  # output tensor shape: (unknown)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        )

        # Define the parallel branch
        parallel_branch = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)  # output tensor shape: (unknown)

        # Instantiate two blocks
        self.block1 = block
        self.block2 = block

        # Instantiate the parallel branch
        self.parallel_branch = parallel_branch

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layer
        self.fc = nn.Linear(in_features='mask', out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Pass input through the first block
        out1 = self.block1(x)  # output tensor shape: (1, 128, 28, 28)

        # Pass input through the second block
        out2 = self.block2(x)  # output tensor shape: (1, 128, 28, 28)

        # Pass input through the parallel branch
        out_parallel = self.parallel_branch(x)  # output tensor shape: (1, 128, 28, 28)

        # Combine the outputs using addition
        combined_out = torch.add(out1, out_parallel)  # output tensor shape: (1, 128, 28, 28)
        combined_out = torch.add(combined_out, out2)  # output tensor shape: (1, 128, 28, 28)
        

        # Concatenate the outputs from the two blocks
        concatenated_out = torch.concat([out1, out2], dim=1)  # output tensor shape: (1, 256, 28, 28)

        # Flatten the concatenated output
        flattened_out = self.flatten(concatenated_out)  # output tensor shape: (1, 200704)

        # Pass the flattened output through the fully connected layer
        output = self.fc(flattened_out)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)