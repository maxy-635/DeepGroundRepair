import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolution to adjust input feature dimensionality to 16
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Basic block
        self.basic_block = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True)
        )

        # Second level residual blocks
        self.residual_block1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True)
        )
        self.residual_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True)
        )
        self.branch_conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Third level global branch
        self.global_branch = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Final layers
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)
        self.fc = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Initial convolution
        x = self.conv1(x)  # output tensor shape: (1, 16, 32, 32)

        # First level (basic block)
        residual = x  # output tensor shape: (1, 16, 32, 32)
        x = self.basic_block(x)  # output tensor shape: (1, 16, 32, 32)
        x = torch.add(x, residual)  # output tensor shape: (1, 16, 32, 32)

        # Second level (residual blocks)
        residual = self.branch_conv1(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.residual_block1(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.residual_block2(x)  # output tensor shape: (1, 32, 32, 32)
        x = torch.add(x, residual)  # output tensor shape: (1, 32, 32, 32)

        # Third level (global branch)
        global_residual = self.global_branch(x)
        x = torch.add(x, global_residual)

        # Final layers
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)