import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )

        # Branch path
        self.branch_path = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Adjust the size based on the output of the main and branch paths
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        main_output = self.main_path(x)  # output tensor shape: (1, 64, 32, 32)
        branch_output = self.branch_path(main_output)  # output tensor shape: (1, 64, 32, 32)

        combined_output = main_output + branch_output  # output tensor shape: (1, 64, 32, 32)

        flattened_output = combined_output.view(1, -1)
        fc_output = self.fc1(flattened_output)
        fc_output = torch.relu(fc_output)
        output = self.fc2(fc_output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)