import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the block that will be used in each branch
        class Block(nn.Module):
            def __init__(self, in_channels, out_channels):
                super(Block, self).__init__()
                self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
                self.depthwise_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels)
                self.conv1x1_2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                residual = x
                out = self.conv1x1_1(x)
                out = self.depthwise_conv(out)
                out = self.conv1x1_2(out)
                out = self.bn1(out)
                out = self.relu(out)
                out += residual
                out = self.bn2(out)
                out = self.relu(out)
                return out

        # Three branches with the same block
        self.branch1 = nn.Sequential(
            Block(1, 32),
            Block(32, 32)
        )
        self.branch2 = nn.Sequential(
            Block(1, 32),
            Block(32, 32)
        )
        self.branch3 = nn.Sequential(
            Block(1, 32),
            Block(32, 32)
        )

        # Concatenate the outputs and classify
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 32 * 28 * 28, 10)  # Assuming input is 28x28 and output channels are 32 per branch

    def forward(self, x):
        # Split the input into three branches
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        # Concatenate the outputs
        out = torch.cat((x1, x2, x3), dim=1)

        # Flatten and classify
        out = self.flatten(out)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)