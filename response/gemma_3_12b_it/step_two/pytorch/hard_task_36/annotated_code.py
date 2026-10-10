import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Pathway
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)

        # Branch Pathway
        self.conv_branch = nn.Conv2d(in_channels=1, out_channels=32, kernel_size='mask', stride='mask', padding='mask')

        # Fusion and Classification
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32 * 7 * 7, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Main Pathway
        x_main = self.conv1(x)  # output tensor shape: (1, 32, 28, 28)
        x_main = F.relu(x_main)  # output tensor shape: (1, 32, 28, 28)
        x_main = self.conv2_1(x_main)  # output tensor shape: (1, 32, 28, 28)
        x_main = F.relu(x_main)  # output tensor shape: (1, 32, 28, 28)
        x_main = self.conv2_2(x_main)  # output tensor shape: (1, 32, 28, 28)
        x_main = F.relu(x_main)  # output tensor shape: (1, 32, 28, 28)
        x_main = self.maxpool(x_main)  # output tensor shape: (1, 32, 14, 14)
        x_main = self.dropout(x_main)  # output tensor shape: (1, 32, 14, 14)

        # Branch Pathway
        x_branch = self.conv_branch(x)  # output tensor shape of this layer should be (1, 32, 14, 14)
        x_branch = F.relu(x_branch)

        # Fusion
        x_fuse = torch.add(x_main, x_branch)

        # Classification
        x_avgpool = self.avgpool(x_fuse)
        x_flatten = self.flatten(x_avgpool)
        x_output = self.linear(x_flatten)

        return x_output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Dummy input tensor (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)