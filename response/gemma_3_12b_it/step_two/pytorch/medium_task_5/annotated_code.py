import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv_block1_main = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.conv_block2_main = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # Branch path
        self.conv_block_branch = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels='mask', kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size='mask', stride='mask', padding='mask')
        )

        # Combination and classification
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=512)  # Adjust input size based on main path output
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=512, out_features=10)


    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        main_path_output = self.conv_block1_main(x)  # output tensor shape: (1, 64, 16, 16)
        main_path_output = self.conv_block2_main(main_path_output)  # output tensor shape: (1, 128, 8, 8)

        # Branch path
        branch_path_output = self.conv_block_branch(x)  # output tensor shape of this layer should be (1, 32, 16, 16)

        # Combine paths
        combined_output = main_path_output + branch_path_output

        # Flatten and classify
        x = self.flatten(combined_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)