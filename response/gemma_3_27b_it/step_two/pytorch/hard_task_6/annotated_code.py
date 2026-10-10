import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_conv1_1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.block1_relu1_1 = nn.ReLU(inplace=True)
        self.block1_conv1_2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.block1_relu1_2 = nn.ReLU(inplace=True)
        self.block1_conv1_3 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.block1_relu1_3 = nn.ReLU(inplace=True)

        # Block 2
        self.block2_groups = 3

        # Block 3
        self.block3_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, groups=1, bias=False)  # Depthwise separable convolution
        self.block3_relu = nn.ReLU(inplace=True)

        # Branch Path
        self.branch_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # Concatenation and Fully Connected Layer
        self.fc = nn.Linear(in_features=10, out_features=10)  # Assuming output size is 10

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main Path
        x1_1 = self.block1_conv1_1(x)  # output tensor shape: (1, 1, 32, 32)
        x1_1 = self.block1_relu1_1(x1_1)  # output tensor shape: (1, 1, 32, 32)
        x1_2 = self.block1_conv1_2(x)  # output tensor shape: (1, 1, 32, 32)
        x1_2 = self.block1_relu1_2(x1_2)  # output tensor shape: (1, 1, 32, 32)
        x1_3 = self.block1_conv1_3(x)  # output tensor shape: (1, 1, 32, 32)
        x1_3 = self.block1_relu1_3(x1_3)  # output tensor shape: (1, 1, 32, 32)
        x_block1 = torch.cat([x1_1, x1_2, x1_3], dim=1)  # output tensor shape: (1, 3, 32, 32)

        # Block 2
        shape = x_block1.shape  # output tensor shape: (unknown)
        height, width, _ = shape[2], shape[3], shape[1]  # output tensor shape: (1); (1); (1)
        x_block2 = torch.reshape(x_block1, (shape[0], height, width, self.block2_groups, 1))  # output tensor shape: (1, 32, 32, 3, 1)
        x_block2 = torch.permute(x_block2, (0, 2, 3, 4, 1))  # output tensor shape: (1, 32, 3, 1, 32)
        x_block2 = torch.reshape('mask', 'mask')

        # Block 3
        x_block3 = self.block3_conv(x_block2)
        x_block3 = self.block3_relu(x_block3)
        
        # Branch Path
        x_branch = self.branch_pool(x)
        x_branch = torch.reshape(x_branch, (x_branch.shape[0], -1))

        # Concatenation
        x_block3 = torch.reshape(x_block3, (x_block3.shape[0], -1))
        x_combined = torch.cat([x_block3, x_branch], dim=1)

        # Fully Connected Layer
        x_output = self.fc(x_combined)

        return x_output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)