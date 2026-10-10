import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the block
        block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.DepthwiseConv2d(in_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        # Create three branches
        self.branch1 = block
        self.branch2 = block
        self.branch3 = block

        # Concatenate the outputs
        self.concat = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=64 * 28 * 28 * 3, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Pass input through each branch
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)

        # Concatenate the outputs
        out = torch.cat((out1, out2, out3), dim=1)

        # Pass through the final layers
        out = self.concat(out)

        return out


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)