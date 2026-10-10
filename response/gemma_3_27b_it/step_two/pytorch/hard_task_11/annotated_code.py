import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main pathway
        self.conv1x1_main = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.relu_main = nn.ReLU(inplace=True)

        # Parallel branch
        self.conv1x1_branch = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv1x3_branch = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(1, 3), stride=1, padding=1)
        self.conv3x1_branch = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 1), stride=1, padding=1)
        self.relu_branch = nn.ReLU(inplace=True)

        # Concatenation and final convolution
        self.concatenate = lambda x, y: torch.concatenate([x, y], dim=1)
        self.conv1x1_final = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.relu_final = nn.ReLU(inplace=True)

        # Direct connection (additive fusion)
        # Fully connected layers for classification
        self.fc1 = nn.Linear(in_features=64 * 10 * 10, out_features=128)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main pathway
        main_path = self.conv1x1_main(x)  # output tensor shape: (1, 64, 32, 32)
        main_path = self.relu_main(main_path)  # output tensor shape: (1, 64, 32, 32)

        # Parallel branch
        branch1 = self.conv1x1_branch(x)  # output tensor shape: (1, 64, 32, 32)
        branch1 = self.relu_branch(branch1)  # output tensor shape: (1, 64, 32, 32)
        branch2 = self.conv1x3_branch(x)  # output tensor shape: (1, 64, 34, 32)
        branch2 = self.relu_branch(branch2)  # output tensor shape: (1, 64, 34, 32)
        branch3 = self.conv3x1_branch(x)  # output tensor shape: (1, 64, 32, 34)
        branch3 = self.relu_branch(branch3)  # output tensor shape: (1, 64, 32, 34)
        branch_path = torch.add(branch1, torch.add(branch2, branch3))

        # Concatenation
        combined_path = self.concatenate(main_path, branch_path)

        # Final convolution
        output = self.conv1x1_final(combined_path)
        output = self.relu_final(output)

        # Additive fusion with input
        output = torch.add(x, output)

        # Flatten for fully connected layers
        output = output.view(-1, 64 * 10 * 10)

        # Fully connected layers
        output = self.fc1(output)
        output = self.relu_fc1(output)
        output = self.fc2(output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)