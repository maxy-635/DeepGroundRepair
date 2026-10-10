import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path_conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)

        # Branch 1
        self.branch1_conv = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Branch 2
        self.branch2_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.branch2_conv = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch2_transpose = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0)

        # Branch 3
        self.branch3_pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        self.branch3_conv = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch3_transpose = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=4, padding=0)

        # Main Path Concatenation
        self.main_path_conv2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Branch Path
        self.branch_path_conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)

        # Fusion
        self.fc = nn.Linear(in_features=16 * 32 * 32, out_features=10) # Assuming input size is 32x32

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main Path
        main_path = self.main_path_conv1(x)  # output tensor shape: (1, 16, 32, 32)

        # Branch 1
        branch1 = self.branch1_conv(main_path)  # output tensor shape: (1, 32, 32, 32)

        # Branch 2
        branch2 = self.branch2_pool(main_path)  # output tensor shape: (1, 16, 16, 16)
        branch2 = self.branch2_conv(branch2)  # output tensor shape: (1, 32, 16, 16)
        branch2 = self.branch2_transpose(branch2)  # output tensor shape: (1, 16, 32, 32)

        # Branch 3
        branch3 = self.branch3_pool(main_path)  # output tensor shape: (1, 16, 8, 8)
        branch3 = self.branch3_conv(branch3)  # output tensor shape: (1, 32, 8, 8)
        branch3 = self.branch3_transpose(branch3)  # output tensor shape: (1, 16, 32, 32)

        # Main Path Concatenation
        main_path_concat = torch.concat([branch1, branch2, branch3], dim=1)  # output tensor shape: (1, 64, 32, 32)
        main_path_output = self.main_path_conv2(main_path_concat)

        # Branch Path
        branch_path = self.branch_path_conv(x)

        # Fusion
        fused = main_path_output + branch_path

        # Fully Connected Layer
        fused = fused.view(fused.size(0), -1)
        output = self.fc(fused)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)