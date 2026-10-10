import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path_initial_conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.main_path_branch1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.main_path_branch2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.main_path_branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.main_path_concat = nn.Conv2d(in_channels=96, out_channels=16, kernel_size=1, stride=1, padding=0)

        # Branch Path
        self.branch_path_initial_conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=16 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)


    def forward(self, x):
        # Main Path
        main_path = F.relu(self.main_path_initial_conv(x))
        branch1_output = F.relu(self.main_path_branch1(main_path))
        branch2_output = self.main_path_branch2(main_path)
        branch3_output = self.main_path_branch3(main_path)
        main_path_output = torch.cat((branch1_output, branch2_output, branch3_output), dim=1)
        main_path_output = F.relu(self.main_path_concat(main_path_output))

        # Branch Path
        branch_path = F.relu(self.branch_path_initial_conv(x))

        # Combine Paths
        combined_output = main_path_output + branch_path

        # Flatten and Fully Connect
        combined_output = combined_output.view(-1, 16 * 32 * 32)
        fc1_output = F.relu(self.fc1(combined_output))
        output = self.fc2(fc1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)