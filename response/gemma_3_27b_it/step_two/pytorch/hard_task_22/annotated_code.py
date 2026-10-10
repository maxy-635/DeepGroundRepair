import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path_group1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.SeparableConv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.SeparableConv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

        self.main_path_group2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.SeparableConv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.SeparableConv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

        self.main_path_group3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.SeparableConv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.SeparableConv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

        # Branch Path
        self.branch_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Fusion and Classification
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=96 * 32 * 32, out_features=128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Split input along the channel dimension
        x_group1 = x[:, :1, :, :]
        x_group2 = x[:, 1:2, :, :]
        x_group3 = x[:, 2:, :, :]

        # Main Path
        main_path_output1 = self.main_path_group1(x_group1)
        main_path_output2 = self.main_path_group2(x_group2)
        main_path_output3 = self.main_path_group3(x_group3)
        main_path_output = torch.cat((main_path_output1, main_path_output2, main_path_output3), dim=1)

        # Branch Path
        branch_path_output = self.branch_path(x)

        # Fusion
        fused_output = main_path_output + branch_path_output

        # Classification
        flattened_output = self.flatten(fused_output)
        fc1_output = self.fc1(flattened_output)
        relu_output = self.relu(fc1_output)
        output = self.fc2(relu_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)