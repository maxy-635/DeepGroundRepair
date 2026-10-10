import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        main_path = nn.Sequential(  # output tensor shape: (unknown)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # Branch path
        branch_path = nn.Sequential(  # output tensor shape: (unknown)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )

        # Combine paths
        self.combined_path = nn.Sequential(
            main_path,
            branch_path
        )

        # Flatten and fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32*8*8 + 16*10*10, out_features=128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Pass input through both paths
        main_output = self.combined_path[0](x)  # output tensor shape: (1, 32, 16, 16)
        branch_output = self.combined_path[1](x)  # output tensor shape: (1, 16, 32, 32)

        # Concatenate the outputs
        combined_output = torch.cat((main_output, branch_output), dim=1)

        # Flatten and pass through fully connected layers
        flattened_output = self.flatten(combined_output)
        fc1_output = self.fc1(flattened_output)
        relu_output = self.relu(fc1_output)
        output = self.fc2(relu_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)