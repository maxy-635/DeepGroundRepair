import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.conv1_main = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_main = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Branch path
        self.conv1_branch = nn.Conv2d(in_channels=3, out_channels=64, kernel_size='mask', stride='mask', padding='mask')

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        main_path = self.conv1_main(x)  # output tensor shape: (1, 32, 32, 32)
        main_path = self.conv2_main(main_path)  # output tensor shape: (1, 64, 32, 32)
        main_path = self.maxpool(main_path)  # output tensor shape: (1, 64, 16, 16)

        # Branch path
        branch_path = self.conv1_branch(x)  # output tensor shape of this layer should be (1, 64, 16, 16)

        # Combine paths
        combined_features = torch.cat((main_path, branch_path), dim=1)
        combined_features = self.flatten(combined_features)
        x = self.fc1(combined_features)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)