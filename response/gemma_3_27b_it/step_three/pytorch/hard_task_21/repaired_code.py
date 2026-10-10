import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32)
        )
        self.main_path_3x3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32)
        )
        self.main_path_5x5 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32)
        )

        # Branch Path
        self.branch_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=96)
        )

        # Classifier
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=98304, out_features=128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main Path
        main_path_1x1_out = self.main_path_1x1(x)  # output tensor shape: (1, 32, 32, 32)
        main_path_3x3_out = self.main_path_3x3(x)  # output tensor shape: (1, 32, 32, 32)
        main_path_5x5_out = self.main_path_5x5(x)  # output tensor shape: (1, 32, 32, 32)

        main_path_out = torch.concat([main_path_1x1_out, main_path_3x3_out, main_path_5x5_out], dim=1)  # output tensor shape: (1, 96, 32, 32)

        # Branch Path
        branch_path_out = self.branch_path(x)  # output tensor shape: (1, 96, 32, 32)

        # Add Main and Branch Paths
        combined_out = main_path_out + branch_path_out  # output tensor shape: (1, 96, 32, 32)

        # Classifier
        flattened_out = self.flatten(combined_out)  # output tensor shape: (1, 98304)
        fc1_out = self.fc1(flattened_out)
        relu_out = self.relu(fc1_out)
        output = self.fc2(relu_out)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)