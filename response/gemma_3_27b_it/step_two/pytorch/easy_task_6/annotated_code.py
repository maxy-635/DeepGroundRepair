import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size='mask', stride='mask', padding='mask')

        # Branch path (direct connection)
        # No layers needed for the branch path initially

        # Flatten and fully connected layer
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=7*7, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Main path
        main_path = self.conv1(x)  # output tensor shape: (1, 8, 14, 14)
        main_path = self.conv2(main_path)  # output tensor shape of this layer should be (1, 1, 28, 28)

        # Branch path
        branch_path = x  # output tensor shape: (1, 1, 28, 28)

        # Combine paths
        combined_path = torch.add(main_path, branch_path)

        # Flatten and fully connected layer
        flattened_path = self.flatten(combined_path)
        output = self.linear(flattened_path)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)