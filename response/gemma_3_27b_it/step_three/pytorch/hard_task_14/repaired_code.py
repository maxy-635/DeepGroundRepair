import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1_main = nn.Linear(in_features=3, out_features=32)
        self.fc2_main = nn.Linear(in_features=32, out_features=32)
        self.relu_main = nn.ReLU(inplace=True)

        # Branch Path
        self.conv_branch = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu_branch = nn.ReLU(inplace=True)

        # Combined Path
        self.fc1_combined = nn.Linear(in_features=32*32*32, out_features=128)
        self.fc2_combined = nn.Linear(in_features=128, out_features=64)
        self.fc3_combined = nn.Linear(in_features=64, out_features=10)
        self.relu_combined = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # the initial shape of x is: (64, 3, 32, 32)
        # Main Path
        main_path = self.global_avg_pool(x)  # output tensor shape: (64, 3, 1, 1)
        main_path = torch.flatten(main_path, start_dim=1)  # output tensor shape: (64, 3)
        main_path = self.fc1_main(main_path)
        main_path = self.relu_main(main_path)
        main_path = self.fc2_main(main_path)
        main_path = torch.reshape(main_path, (x.size(0), x.size(1), 1, 1))

        # Branch Path
        branch_path = self.conv_branch(x)
        branch_path = self.relu_branch(branch_path)

        # Combine Paths
        combined_path = torch.mul(branch_path, main_path)

        # Fully Connected Layers
        combined_path = torch.flatten(combined_path, start_dim=1)
        combined_path = self.fc1_combined(combined_path)
        combined_path = self.relu_combined(combined_path)
        combined_path = self.fc2_combined(combined_path)
        combined_path = self.relu_combined(combined_path)
        combined_path = self.fc3_combined(combined_path)
        output = self.softmax(combined_path)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(64, 3, 32, 32)  # Batch size of 64, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [64, 10]