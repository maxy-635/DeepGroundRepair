import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the block
        block = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=7, padding=0),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=3),
            nn.Sigmoid()
        )

        # Create two branches with the same block
        self.branch1 = block
        self.branch2 = block

        # Concatenate the outputs of the branches
        self.concat = nn.Concat(dim=1)

        # Flatten the concatenated output
        self.flatten = nn.Flatten()

        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=3 * 32 * 10 * 10, out_features=10)

    def forward(self, x):
        # Pass input through both branches
        out1 = self.branch1(x)
        out2 = self.branch2(x)

        # Concatenate the outputs
        out = self.concat( [out1, out2] )

        # Flatten the concatenated output
        out = self.flatten(out)

        # Pass through the fully connected layer
        out = self.fc(out)

        return out


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)