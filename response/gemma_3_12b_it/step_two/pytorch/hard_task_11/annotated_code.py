import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1x1_main = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv1x1_branch1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv1x3_branch1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3x1_branch1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1x1_fuse = nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')
        self.fc1 = nn.Linear(in_features=32 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main pathway
        main_path = self.conv1x1_main(x)  # output tensor shape: (1, 32, 32, 32)

        # Branch pathway
        branch1_1x1 = self.conv1x1_branch1(x)  # output tensor shape: (1, 32, 32, 32)
        branch1_1x3 = self.conv1x3_branch1(x)  # output tensor shape: (1, 32, 32, 32)
        branch1_3x1 = self.conv3x1_branch1(x)  # output tensor shape: (1, 32, 32, 32)
        branch_concat = torch.cat((branch1_1x1, branch1_1x3, branch1_3x1), dim=1)  # output tensor shape: (1, 96, 32, 32)
        branch_path = self.conv1x1_fuse(branch_concat)

        # Fusion
        combined_path = torch.add(main_path, branch_path)

        # Flatten and fully connected layers
        flattened = combined_path.view(x.size(0), -1)
        fc1_output = F.relu(self.fc1(flattened))
        output = self.fc2(fc1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)