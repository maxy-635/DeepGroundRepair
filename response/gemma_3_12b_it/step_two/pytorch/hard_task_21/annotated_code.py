import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path = nn.Sequential(
            # Split into three groups along the channel
            # Group 1: 1x1 kernel
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            # Group 2: 3x3 kernel
            nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask'),
            # Group 3: 5x5 kernel
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        )

        # Concatenate the outputs from the three groups
        self.concat = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        )
        

        # Branch Path
        self.branch_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        )

        # Add the outputs from both paths
        self.add = nn.Sequential()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main Path
        main_path_output = self.main_path(x)
        main_path_output = self.concat(main_path_output)

        # Branch Path
        branch_path_output = self.branch_path(x)

        # Add the outputs
        added_output = torch.add(main_path_output, branch_path_output)

        # Flatten the output for fully connected layers
        flattened_output = added_output.view(added_output.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(flattened_output))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)