import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # Branch Path
        self.branch_path = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=7, padding=0),
            nn.Flatten(),
            nn.Linear(in_features='mask', out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=32)
        )

        # Final Classification Layers
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main Path
        main_output = self.main_path(x)  # output tensor shape: (1, 64, 16, 16)

        # Branch Path
        branch_output = self.branch_path(x)
        branch_output = torch.reshape(branch_output, (branch_output.shape[0], 1, 1, 32))
        branch_output = branch_output.repeat(1, 64, 8, 8)

        # Combine Paths
        combined_output = torch.mul(main_output, branch_output) + main_output

        # Classification
        combined_output = torch.flatten(combined_output, 1)
        fc1_output = self.fc1(combined_output)
        fc1_output = F.relu(fc1_output)
        output = self.fc2(fc1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)