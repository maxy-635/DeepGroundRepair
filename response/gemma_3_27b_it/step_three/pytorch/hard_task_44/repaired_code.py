import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.block1_conv2 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.block1_conv3 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.block1_dropout = nn.Dropout(p=0.2)

        # Block 2
        self.block2_branch1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.block2_branch2_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.block2_branch2_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.block2_branch3_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.block2_branch3_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.block2_branch4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.block2_branch4_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)


        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=131072, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1
        out1_1 = self.block1_conv1(x)  # output tensor shape: (1, 32, 32, 32)
        out1_2 = self.block1_conv2(x)  # output tensor shape: (1, 32, 32, 32)
        out1_3 = self.block1_conv3(x)  # output tensor shape: (1, 32, 32, 32)
        out1_concat = torch.cat([out1_1, out1_2, out1_3], dim=1)  # output tensor shape: (1, 96, 32, 32)
        out1_dropout = self.block1_dropout(out1_concat)  # output tensor shape: (1, 96, 32, 32)

        # Block 2
        out2_1 = self.block2_branch1(x)  # output tensor shape: (1, 32, 32, 32)
        out2_2 = self.block2_branch2_conv2(self.block2_branch2_conv1(x))  # output tensor shape: (1, 32, 32, 32)
        out2_3 = self.block2_branch3_conv2(self.block2_branch3_conv1(x))  # output tensor shape: (1, 32, 32, 32)
        out2_4 = self.block2_branch4_conv(self.block2_branch4_pool(x))  # output tensor shape: (1, 32, 32, 32)
        out2_concat = torch.cat([out2_1, out2_2, out2_3, out2_4], dim=1)  # output tensor shape: (1, 128, 32, 32)

        # Flatten and Linear
        flattened = self.flatten(out2_concat)  # output tensor shape: (1, 131072)
        output = self.linear(flattened)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)