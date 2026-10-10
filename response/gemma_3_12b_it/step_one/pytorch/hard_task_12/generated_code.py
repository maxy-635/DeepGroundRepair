import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main Path
        self.conv1x1_main = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv1x1_feature = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv3x3_feature = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=16 * 32 * 32, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=10)  # Assuming 10 classes for classification

        # Branch Path
        self.conv3x3_branch = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Main Path
        main_path = self.relu(self.conv1x1_main(x))
        main_path_1 = self.relu(self.conv1x1_feature(main_path))
        main_path_2 = self.relu(self.conv3x3_feature(main_path))
        main_path_combined = torch.cat((main_path_1, main_path_2), dim=1)

        # Branch Path
        branch_path = self.relu(self.conv3x3_branch(x))

        # Combine Main and Branch Paths
        combined_path = torch.add(main_path_combined, branch_path)

        # Classification
        flattened = self.flatten(combined_path)
        fc1_output = self.relu(self.fc1(flattened))
        output = self.fc2(fc1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 64, 32, 32)  # Batch size 1, 64 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)