import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First level: Basic block
        self.basic_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Second level: Two residual blocks
        self.residual_block1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.residual_block2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.branch_conv1 = nn.Conv2d(16, 16, kernel_size=1, stride=1)

        # Third level: Global branch
        self.global_branch_conv = nn.Conv2d(16, 16, kernel_size=1, stride=1)

        # Additions for residual connections
        self.add1 = nn.Sequential()
        self.add2 = nn.Sequential()
        self.add3 = nn.Sequential()

        # Fully connected layers for classification
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First level
        x1 = self.basic_block(x)  # output tensor shape: (1, 16, 32, 32)

        # Second level
        x2 = self.basic_block(x1)
        branch_output = self.branch_conv1(x1)
        x2 = torch.add(x2, branch_output)

        # Third level
        global_branch_output = self.global_branch_conv(x1)
        x3 = torch.add(x2, global_branch_output)

        # Classification
        x4 = self.avg_pool(x3)
        x4 = torch.flatten(x4, 1)
        output = self.fc(x4)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)